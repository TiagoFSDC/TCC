import cv2
import torch
import threading
import time
import numpy as np
from collections import deque
import pandas as pd  # Added explicit import for pandas
import warnings
import sys
import os

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

class SmartTrafficLight:
    def __init__(self, camera1_source=0, camera2_source=1):
        # Configurações do semáforo (valores base)
        self.GREEN_TIME = 25  # Adjusted to be between MIN and MAX
        # self.RED_TIME = 30
        self.YELLOW_TIME = 3
        self.EMERGENCY_RED_TIME = 8.3

        # Parâmetros para cálculo de verde mínimo (Equação 8.2)
        self.t_pin = 3.0  # Tempo perdido no início, em segundos
        self.d = 20.0  # Distância entre linha de retenção e seção de detecção, em metros
        self.esp = 6.0  # Espaçamento médio entre frentes dos automóveis em fila, em metros
        self.FS = 1800  # Fluxo de saturação (veículos/hora) - valor típico para via urbana
        self.i_fs = 3600 / self.FS  # Intervalo entre veículos (segundos)
        
        # Cálculo do verde mínimo: tv,min = tpin + (d/esp) + ifs
        self.GREEN_MIN = self.t_pin + (self.d / self.esp) + self.i_fs
        
        # Parâmetros para cálculo de verde máximo (Equação 8.10)
        self.t_c_fixo = 60.0  # Tempo de ciclo para operação em tempo fixo, em segundos
        self.t_c = 1.4 * self.t_c_fixo  # tc = 1,4 × tc,fixo
        
        # Verde máximo é uma fração do tempo de ciclo (considerando 2 fases)
        # Subtraindo tempos de amarelo e vermelho total
        self.GREEN_MAX = (self.t_c / 2) - self.YELLOW_TIME
        
        # Extensão de verde baseada no intervalo entre veículos
        self.GAP_EXTENSION = self.i_fs * 1.5  # Multiplicador para dar margem de detecção

        print(f"⚙️ Parâmetros MBST Vol. V:")
        print(f"   Verde mínimo (Eq. 8.2): {self.GREEN_MIN:.1f}s")
        print(f"   Verde máximo (Eq. 8.10): {self.GREEN_MAX:.1f}s")
        print(f"   Extensão de verde: {self.GAP_EXTENSION:.1f}s")
        print(f"   Tempo de ciclo atuado: {self.t_c:.1f}s")

        # Estados do semáforo
        self.STATES = {'GREEN': 'VERDE', 'YELLOW': 'AMARELO', 'RED': 'VERMELHO'}
        self.detection_history = {'A': deque(maxlen=10), 'B': deque(maxlen=10)}

        self.semaphore_A_state = 'RED'
        self.semaphore_B_state = 'GREEN'

        # duração planejada do estágio atual
        self.phase_duration = self.GREEN_TIME
        self.state_start_time = time.time()
        self.green_end_time = self.state_start_time + self.phase_duration
        
        self.extensions_applied = 0
        self.last_extension_time = 0
        self.can_extend = True  # Flag para controlar se ainda pode estender

        self.running = True
        self.emergency_activated = False
        self.emergency_target = None
        self.cars_detected = {'A': False, 'B': False}

        self.class_names = {2: 'Carro', 3: 'Moto', 5: 'Onibus', 7: 'Caminhao'}
        self.class_colors = {2: (0, 255, 0), 3: (255, 0, 255), 5: (0, 165, 255), 7: (0, 0, 255)}

        self.bg_subtractor_A = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=25, detectShadows=False)
        self.bg_subtractor_B = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=25, detectShadows=False)

        self.load_yolo_model()

        self.camera1 = cv2.VideoCapture(camera1_source)
        self.camera2 = cv2.VideoCapture(camera2_source)
        self.setup_cameras()

        self.detection_thread = None
        self.control_thread = None

    def load_yolo_model(self):
        try:
            # Tenta primeiro usar o repositório local se existir
            yolo_repo = '/opt/yolov5'
            hubconf_path = os.path.join(yolo_repo, 'hubconf.py')
            
            if os.path.exists(hubconf_path):
                # Usa repositório local se disponível
                yolo_repo_normalized = os.path.normpath(yolo_repo)
                self.yolo_model = torch.hub.load(yolo_repo_normalized, 'yolov5s', source='local', pretrained=True)
            else:
                # Fallback: usa ultralytics via torch.hub (baixa automaticamente)
                self.yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
            
            self.yolo_model.conf = 0.35  # Reduzido ainda mais para detectar motos
            self.yolo_model.iou = 0.4
            # Incluindo classe 3 (motorcycle) na lista
            self.vehicle_classes = [2, 3, 5, 7]  # car, motorcycle, bus, truck
            print("✔ Modelo YOLOv5 carregado com sucesso")
            print("✔ Classes de veículos detectadas: Carro(2), Moto(3), Ônibus(5), Caminhão(7)")
        except Exception as e:
            print(f"❌ Erro ao carregar YOLOv5: {e}")
            import traceback
            traceback.print_exc()

    def setup_cameras(self):
        for camera in [self.camera1, self.camera2]:
            camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
            camera.set(cv2.CAP_PROP_FPS, 30)

    def detect_vehicles(self, frame, fgmask, street):
        try:
            results = self.yolo_model(frame)
            detections = results.pandas().xyxy[0]
            
            vehicles = detections[detections['class'].isin(self.vehicle_classes)]
            
            # Tratamento especial para motos (classe 3) - aceitar confiança menor
            motorcycles = vehicles[(vehicles['class'] == 3) & (vehicles['confidence'] >= 0.25)]
            other_vehicles = vehicles[(vehicles['class'].isin([2, 5, 7])) & (vehicles['confidence'] >= 0.35)]
            
            # Combinar detecções
            vehicles = pd.concat([motorcycles, other_vehicles])
            
            moving_count = 0

            # Desenhar bounding boxes e rótulos (desenhar todos, mas contar apenas em movimento)
            for _, detection in vehicles.iterrows():
                x1, y1, x2, y2 = int(detection['xmin']), int(detection['ymin']), int(detection['xmax']), int(detection['ymax'])
                class_id = int(detection['class'])
                confidence = detection['confidence']
                class_name = self.class_names.get(class_id, f'Veículo({class_id})')
                color = self.class_colors.get(class_id, (255, 255, 255))
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                label = f"{class_name}: {confidence:.2f}"
                cv2.putText(frame, label, (x1, max(15, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

                # Verificar se está em movimento
                if fgmask is not None:
                    bbox_mask = fgmask[y1:y2, x1:x2]
                    fg_pixels = cv2.countNonZero(bbox_mask)
                    bbox_area = (x2 - x1) * (y2 - y1)
                    if fg_pixels > 0.05 * bbox_area:  # 5% de pixels em movimento
                        moving_count += 1

            return moving_count > 0, frame
        except Exception as e:
            print(f"Erro na detecção: {e}")
            return False, frame

    def stabilize_detection(self, street, detected):
        self.detection_history[street].append(detected)
        positive_detections = sum(self.detection_history[street])
        total_detections = len(self.detection_history[street])
        return (positive_detections / total_detections) >= 0.4 if total_detections > 0 else False

    def detection_loop(self):
        while self.running:
            try:
                ret1, frame1 = self.camera1.read()
                ret2, frame2 = self.camera2.read()

                if ret1 and ret2:
                    frame1 = cv2.resize(frame1, (640, 360))
                    frame2 = cv2.resize(frame2, (640, 360))

                    fgmask1 = self.bg_subtractor_A.apply(frame1)
                    fgmask2 = self.bg_subtractor_B.apply(frame2)

                    cars_A, annotated_frame1 = self.detect_vehicles(frame1, fgmask1, 'A')
                    cars_B, annotated_frame2 = self.detect_vehicles(frame2, fgmask2, 'B')

                    # Atualizar detecções estabilizadas
                    self.cars_detected['A'] = self.stabilize_detection('A', cars_A)
                    self.cars_detected['B'] = self.stabilize_detection('B', cars_B)

                    # Exibir frames
                    cv2.imshow('RUA A', annotated_frame1)
                    cv2.imshow('RUA B', annotated_frame2)

                    # Atualizar janela de status
                    self.create_status_window()

                # Usar única chamada waitKey aqui (centralizado)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    self.running = False

            except Exception as e:
                print(f"Erro no loop de detecção: {e}")

    def draw_traffic_light(self, img, pos, state):
        # Posições dos círculos
        red_pos = (pos[0], pos[1])
        yellow_pos = (pos[0], pos[1] + 40)
        green_pos = (pos[0], pos[1] + 80)

        # Cores off
        off_color = (50, 50, 50)

        # Red
        color = (0, 0, 255) if state == 'RED' else off_color
        cv2.circle(img, red_pos, 15, color, -1)

        # Yellow
        color = (0, 255, 255) if state == 'YELLOW' else off_color
        cv2.circle(img, yellow_pos, 15, color, -1)

        # Green
        color = (0, 255, 0) if state == 'GREEN' else off_color
        cv2.circle(img, green_pos, 15, color, -1)

    def create_status_window(self):
        status_img = np.full((350, 500, 3), 30, dtype=np.uint8)

        cv2.putText(status_img, "SISTEMA DE SEMAFORO INTELIGENTE", (50, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (230, 230, 230), 2)
        cv2.line(status_img, (20, 40), (480, 40), (80, 80, 80), 1)

        current_time = time.time()
        if self.semaphore_A_state == 'GREEN' or self.semaphore_B_state == 'GREEN':
            remaining = max(0.0, self.green_end_time - current_time)
        else:
            elapsed = current_time - self.state_start_time
            remaining = max(0.0, self.phase_duration - elapsed)

        # Estados dos semáforos com cores
        color_A = (0, 255, 0) if self.semaphore_A_state == 'GREEN' else (0, 200, 200) if self.semaphore_A_state == 'YELLOW' else (0, 0, 200)
        color_B = (0, 255, 0) if self.semaphore_B_state == 'GREEN' else (0, 200, 200) if self.semaphore_B_state == 'YELLOW' else (0, 0, 200)

        cv2.putText(status_img, f"RUA A: Semaforo: {self.STATES[self.semaphore_A_state]} Veiculos: {'Sim' if self.cars_detected['A'] else 'Nao'}", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_A, 1)
        cv2.putText(status_img, f"RUA B: Semaforo: {self.STATES[self.semaphore_B_state]} Veiculos: {'Sim' if self.cars_detected['B'] else 'Nao'}", (30, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_B, 1)
        
        cv2.putText(status_img, f"TIMER: {remaining:.1f} segundos", (30, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 220, 0), 2)

        # Desenhar semáforos gráficos
        self.draw_traffic_light(status_img, (100, 170), self.semaphore_A_state)
        self.draw_traffic_light(status_img, (300, 170), self.semaphore_B_state)

        cv2.putText(status_img, "RUA A", (70, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)  # Adjusted y-position
        cv2.putText(status_img, "RUA B", (270, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)  # Adjusted y-position

        if self.extensions_applied > 0:
            cv2.putText(status_img, f"Extensoes aplicadas: {self.extensions_applied}", (30, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

        # Status operacional simplificado
        if self.emergency_activated:
            cv2.putText(status_img, f"EMERGENCIA - RUA {self.emergency_target}", (30, 320), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        elif self.can_extend and ((self.semaphore_A_state == 'GREEN' and self.cars_detected['A'] and not self.cars_detected['B']) or 
                                 (self.semaphore_B_state == 'GREEN' and self.cars_detected['B'] and not self.cars_detected['A'])):
            cv2.putText(status_img, "Extensao de verde ativa", (30, 320), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 255), 1)

        cv2.putText(status_img, "Pressione 'q' nas janelas das cameras para sair", (30, 340), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

        cv2.imshow('Status do Sistema', status_img)

    def activate_emergency_green(self, street):
        if self.emergency_activated:
            return

        current_time = time.time()
        remaining = self.green_end_time - current_time

        # Reduzir o tempo restante para EMERGENCY_RED_TIME
        if remaining > self.EMERGENCY_RED_TIME:
            self.green_end_time = current_time + self.EMERGENCY_RED_TIME

        self.emergency_activated = True
        self.emergency_target = street
        print(f"🚨 EMERGÊNCIA: solicitando verde para a rua {street} (remaining reduzido para {max(0.0, self.green_end_time - current_time):.1f}s)")

    def apply_intelligent_logic(self):
        cars_A = self.cars_detected['A']
        cars_B = self.cars_detected['B']
        current_time = time.time()
        elapsed = current_time - self.state_start_time
        remaining = self.green_end_time - current_time

        # 🚨 Emergência - apenas quando há carros em uma rua e nenhum na outra e após o verde mínimo
        if elapsed >= self.GREEN_MIN:  # Verificar se o verde mínimo foi respeitado
            if cars_A and not cars_B:
                if self.semaphore_A_state == 'RED' and self.semaphore_B_state == 'GREEN' and not self.emergency_activated:
                    self.activate_emergency_green('A')

            if cars_B and not cars_A:
                if self.semaphore_B_state == 'RED' and self.semaphore_A_state == 'GREEN' and not self.emergency_activated:
                    self.activate_emergency_green('B')

        # Aplicar extensão quando:
        # 1. Semáforo está verde para a rua
        # 2. Há veículos APENAS nessa rua (não na outra)
        # 3. Não estamos em modo emergência
        # 4. Ainda podemos estender (não excedemos verde máximo)
        # 5. Tempo desde última extensão > GAP_EXTENSION/2 (evitar extensões muito frequentes)
        
        if not self.emergency_activated and self.can_extend:
            extension_cooldown = self.GAP_EXTENSION / 2
            
            current_duration = self.green_end_time - self.state_start_time
            
            # Extensão para rua A
            if (self.semaphore_A_state == 'GREEN' and cars_A and not cars_B and 
                current_duration < self.GREEN_MAX and 
                (current_time - self.last_extension_time) > extension_cooldown):
                
                # Aplicar extensão quando restam poucos segundos
                if remaining <= self.GAP_EXTENSION:
                    extension = min(self.GAP_EXTENSION, self.GREEN_MAX - current_duration)
                    if extension > 0.5:  # Só aplicar se extensão for significativa
                        self.green_end_time += extension
                        self.extensions_applied += 1
                        self.last_extension_time = current_time
                        print(f"➕ Extensão aplicada em A (+{extension:.1f}s) - Total: {self.green_end_time - self.state_start_time:.1f}s")
                        
                        # Verificar se atingiu o máximo
                        if self.green_end_time - self.state_start_time >= self.GREEN_MAX:
                            self.can_extend = False
                            print(f"🔴 Verde máximo atingido em A ({self.GREEN_MAX:.1f}s)")

            # Extensão para rua B
            elif (self.semaphore_B_state == 'GREEN' and cars_B and not cars_A and 
                  current_duration < self.GREEN_MAX and 
                  (current_time - self.last_extension_time) > extension_cooldown):
                
                # Aplicar extensão quando restam poucos segundos
                if remaining <= self.GAP_EXTENSION:
                    extension = min(self.GAP_EXTENSION, self.GREEN_MAX - current_duration)
                    if extension > 0.5:  # Só aplicar se extensão for significativa
                        self.green_end_time += extension
                        self.extensions_applied += 1
                        self.last_extension_time = current_time
                        print(f"➕ Extensão aplicada em B (+{extension:.1f}s) - Total: {self.green_end_time - self.state_start_time:.1f}s")
                        
                        # Verificar se atingiu o máximo
                        if self.green_end_time - self.state_start_time >= self.GREEN_MAX:
                            self.can_extend = False
                            print(f"🔴 Verde máximo atingido em B ({self.GREEN_MAX:.1f}s)")

    def control_traffic_lights(self):
        while self.running:
            try:
                current_time = time.time()
                elapsed_time = current_time - self.state_start_time

                is_green_phase = self.semaphore_A_state == 'GREEN' or self.semaphore_B_state == 'GREEN'

                if is_green_phase:
                    self.apply_intelligent_logic()
                    if current_time >= self.green_end_time:
                        self.transition_state()
                else:
                    if elapsed_time >= self.phase_duration:
                        self.transition_state()

                time.sleep(0.05)
            except Exception as e:
                print(f"Erro no controle de semáforos: {e}")

    def transition_state(self):
        current_time = time.time()
        cars_A = self.cars_detected['A']
        cars_B = self.cars_detected['B']

        if self.semaphore_A_state == 'GREEN':
            # A verde -> A amarelo
            self.semaphore_A_state = 'YELLOW'
            self.semaphore_B_state = 'RED'
            self.phase_duration = self.YELLOW_TIME
            self.extensions_applied = 0
            self.can_extend = True

        elif self.semaphore_A_state == 'YELLOW':
            # A amarelo -> B verde
            self.semaphore_A_state = 'RED'
            self.semaphore_B_state = 'GREEN'
            
            if self.emergency_activated and self.emergency_target == 'A':
                # Emergência foi para A, agora B fica verde
                if not cars_B:
                    # B não tem carros, aplicar verde mínimo
                    self.phase_duration = self.GREEN_MIN
                    print(f"🔧 Verde mínimo aplicado para B (sem veículos após emergência): {self.GREEN_MIN:.1f}s")
                else:
                    # B tem carros, usar tempo normal
                    self.phase_duration = self.GREEN_TIME
            else:
                # Transição normal
                self.phase_duration = max(self.GREEN_MIN, self.GREEN_TIME)
            self.green_end_time = current_time + self.phase_duration

        elif self.semaphore_A_state == 'RED' and self.semaphore_B_state == 'GREEN':
            # B verde -> B amarelo
            self.semaphore_A_state = 'RED'
            self.semaphore_B_state = 'YELLOW'
            self.phase_duration = self.YELLOW_TIME
            self.extensions_applied = 0
            self.can_extend = True

        elif self.semaphore_B_state == 'YELLOW':
            # B amarelo -> A verde
            self.semaphore_B_state = 'RED'
            self.semaphore_A_state = 'GREEN'
            
            if self.emergency_activated and self.emergency_target == 'B':
                # Emergência foi para B, agora A fica verde
                if not cars_A:
                    # A não tem carros, aplicar verde mínimo
                    self.phase_duration = self.GREEN_MIN
                    print(f"🔧 Verde mínimo aplicado para A (sem veículos após emergência): {self.GREEN_MIN:.1f}s")
                else:
                    # A tem carros, usar tempo normal
                    self.phase_duration = self.GREEN_TIME
            else:
                # Transição normal
                self.phase_duration = max(self.GREEN_MIN, self.GREEN_TIME)
            self.green_end_time = current_time + self.phase_duration

        # Resetar timer e flags de emergência ao completar a transição
        self.state_start_time = current_time
        if self.emergency_activated:
            print(f"✅ Emergência atendida (target={self.emergency_target})")
        self.emergency_activated = False
        self.emergency_target = None

        print(f"🚦 Transição: Rua A = {self.STATES[self.semaphore_A_state]}, Rua B = {self.STATES[self.semaphore_B_state]} | Duração: {self.phase_duration:.1f}s")

    def start(self):
        print("🚀 Iniciando Sistema de Semaforo Inteligente")
        print("📋 Deteccao de veiculos: Carros, Motos, Onibus e Caminhoes")
        print("📖 Baseado nas normas MBST Vol. V - Sinalização Semafórica")
        self.detection_thread = threading.Thread(target=self.detection_loop)
        self.control_thread = threading.Thread(target=self.control_traffic_lights)
        self.detection_thread.daemon = True
        self.control_thread.daemon = True
        self.detection_thread.start()
        self.control_thread.start()

        try:
            while self.running:
                time.sleep(1)
        except KeyboardInterrupt:
            self.stop()

    def stop(self):
        print("🛑 Parando sistema...")
        self.running = False
        if self.camera1:
            self.camera1.release()
        if self.camera2:
            self.camera2.release()
        cv2.destroyAllWindows()
        print("✔ Sistema parado")

def main():
    traffic_system = SmartTrafficLight('video1.mp4', 'IMG_2268.mp4')
    try:
        traffic_system.start()
    except Exception as e:
        print(f"❌ Erro: {e}")
    finally:
        traffic_system.stop()

if __name__ == "__main__":
    main()