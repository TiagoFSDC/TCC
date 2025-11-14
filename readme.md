# 🚦 Sistema de Semáforo Inteligente

Sistema inteligente de controle de semáforos baseado em visão computacional e detecção de veículos em tempo real, implementado conforme as normas **MBST Vol. V - Sinalização Semafórica**.

> ⚠️ **AVISO IMPORTANTE**: Este código foi desenvolvido para Python 3.8 até Python 3.11. **Versões acima de Python 3.11 não são compatíveis** e o sistema não funcionará corretamente. Por favor, use Python 3.8, 3.9, 3.10 ou 3.11.

## 📋 Índice

- [Sobre o Projeto](#sobre-o-projeto)
- [Funcionalidades](#funcionalidades)
- [Requisitos](#requisitos)
- [Instalação](#instalação)
- [Como Usar](#como-usar)
- [Estrutura do Código](#estrutura-do-código)
- [Parâmetros Técnicos](#parâmetros-técnicos)
- [Tecnologias Utilizadas](#tecnologias-utilizadas)
- [Exemplos de Uso](#exemplos-de-uso)
- [Troubleshooting](#troubleshooting)

## 🎯 Sobre o Projeto

Este projeto implementa um sistema de controle inteligente de semáforos que utiliza:
- **YOLOv5** para detecção de veículos (carros, motos, ônibus e caminhões)
- **Background Subtraction** para identificar veículos em movimento
- **Lógica adaptativa** baseada nas normas MBST Vol. V para otimizar os tempos de sinalização
- **Controle em tempo real** de dois semáforos sincronizados (Rua A e Rua B)

O sistema ajusta automaticamente os tempos de verde baseado no fluxo de veículos detectado, aplicando extensões de verde quando necessário e respeitando os limites mínimo e máximo definidos pelas normas técnicas.

## ✨ Funcionalidades

### Detecção de Veículos
- ✅ Detecção de **carros**, **motos**, **ônibus** e **caminhões** usando YOLOv5
- ✅ Identificação de veículos em movimento através de subtração de fundo
- ✅ Estabilização de detecções para evitar falsos positivos
- ✅ Visualização em tempo real com bounding boxes e labels

### Controle Inteligente
- ✅ **Extensão de verde**: Estende o tempo de verde quando há veículos apenas em uma rua
- ✅ **Modo emergência**: Reduz o tempo de vermelho quando uma rua tem veículos e a outra está vazia
- ✅ **Respeito aos limites**: Garante verde mínimo e máximo conforme MBST Vol. V
- ✅ **Transições suaves**: Gerenciamento automático de fases (verde → amarelo → vermelho)

### Interface Visual
- ✅ Janelas de visualização para cada câmera (Rua A e Rua B)
- ✅ Janela de status com informações em tempo real:
  - Estado atual dos semáforos
  - Presença de veículos
  - Timer de contagem regressiva
  - Indicadores de extensão e emergência
  - Representação gráfica dos semáforos

## 📦 Requisitos

### Hardware
- **Câmeras**: 2 câmeras USB ou arquivos de vídeo
- **GPU** (recomendado): NVIDIA GPU com suporte CUDA para melhor performance
- **RAM**: Mínimo 8GB (recomendado 16GB)
- **Processador**: CPU multi-core recomendado

### Software
- **Python 3.8, 3.9, 3.10 ou 3.11** (⚠️ **NÃO use Python 3.12 ou superior** - não é compatível)
- OpenCV (cv2)
- PyTorch
- YOLOv5 (via torch.hub)
- NumPy
- Pandas

## 🔧 Instalação

### 0. Verifique a versão do Python

⚠️ **IMPORTANTE**: Antes de prosseguir, verifique se você está usando Python 3.8, 3.9, 3.10 ou 3.11:

```bash
python --version
```

Se você tiver Python 3.12 ou superior, será necessário instalar uma versão compatível. O código **não funcionará** com versões acima de Python 3.11.

### 1. Clone ou baixe o repositório

```bash
cd c:\Users\tiago\Desktop\Trabalho-Facul\TCC\TCC\TCC
```

### 2. Instale as dependências

```bash
pip install opencv-python torch torchvision numpy pandas ultralytics
```

**Nota**: Para melhor performance com GPU, instale PyTorch com suporte CUDA:
```bash
# Para CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### 3. Verifique a instalação

```bash
python -c "import cv2, torch; print('OpenCV:', cv2.__version__); print('PyTorch:', torch.__version__)"
```

## 🚀 Como Usar

### Uso Básico com Câmeras USB

```python
from TCC import SmartTrafficLight

# Inicializar com câmeras USB (0 e 1)
traffic_system = SmartTrafficLight(camera1_source=0, camera2_source=1)

try:
    traffic_system.start()
except KeyboardInterrupt:
    traffic_system.stop()
```

### Uso com Arquivos de Vídeo

```python
from TCC import SmartTrafficLight

# Inicializar com arquivos de vídeo
traffic_system = SmartTrafficLight('video1.mp4', 'video2.mp4')

try:
    traffic_system.start()
except KeyboardInterrupt:
    traffic_system.stop()
```

### Executar o Script Principal

```bash
python TCC.py
```

### Controles
- **Pressione 'q'** em qualquer janela de câmera para encerrar o sistema
- **Ctrl+C** no terminal também encerra o sistema

## 📁 Estrutura do Código

### Classe `SmartTrafficLight`

#### Métodos Principais

| Método | Descrição |
|--------|-----------|
| `__init__()` | Inicializa o sistema, carrega modelo YOLO e configura câmeras |
| `load_yolo_model()` | Carrega o modelo YOLOv5 pré-treinado |
| `detect_vehicles()` | Detecta veículos em um frame usando YOLO e verifica movimento |
| `stabilize_detection()` | Estabiliza detecções usando histórico (evita falsos positivos) |
| `detection_loop()` | Loop principal de detecção (executa em thread separada) |
| `control_traffic_lights()` | Loop de controle dos semáforos (executa em thread separada) |
| `apply_intelligent_logic()` | Aplica lógica de extensão de verde e emergência |
| `transition_state()` | Gerencia transições entre estados dos semáforos |
| `create_status_window()` | Cria e atualiza janela de status |
| `start()` | Inicia o sistema (inicia threads) |
| `stop()` | Para o sistema e libera recursos |

#### Estados do Semáforo
- `GREEN` (VERDE): Semáforo aberto para tráfego
- `YELLOW` (AMARELO): Fase de transição
- `RED` (VERMELHO): Semáforo fechado

## 📊 Parâmetros Técnicos

### Parâmetros Base (MBST Vol. V)

| Parâmetro | Valor | Descrição |
|-----------|-------|-----------|
| `GREEN_TIME` | 25s | Tempo base de verde |
| `YELLOW_TIME` | 3s | Tempo de amarelo |
| `EMERGENCY_RED_TIME` | 8.3s | Tempo máximo de vermelho em emergência |

### Cálculo de Verde Mínimo (Equação 8.2)

```
tv,min = tpin + (d/esp) + ifs
```

Onde:
- `t_pin` = 3.0s (tempo perdido no início)
- `d` = 20.0m (distância linha de retenção → detecção)
- `esp` = 6.0m (espaçamento médio entre veículos)
- `FS` = 1800 veículos/hora (fluxo de saturação)
- `i_fs` = 3600/FS = 2.0s (intervalo entre veículos)

**Resultado**: `GREEN_MIN ≈ 8.3s`

### Cálculo de Verde Máximo (Equação 8.10)

```
tc = 1.4 × tc,fixo
GREEN_MAX = (tc / 2) - YELLOW_TIME
```

Onde:
- `t_c_fixo` = 60.0s (tempo de ciclo fixo)
- `t_c` = 84.0s (tempo de ciclo atuado)
- `GREEN_MAX ≈ 39.0s`

### Extensão de Verde

- `GAP_EXTENSION` = `i_fs × 1.5 ≈ 3.0s`
- Aplicada quando há veículos apenas em uma rua
- Respeita o limite máximo de verde
- Cooldown de `GAP_EXTENSION/2` entre extensões

## 🛠️ Tecnologias Utilizadas

- **OpenCV**: Processamento de imagem e vídeo
- **PyTorch**: Framework de deep learning
- **YOLOv5**: Modelo de detecção de objetos
- **NumPy**: Operações numéricas
- **Pandas**: Manipulação de dados de detecção
- **Threading**: Processamento paralelo (detecção + controle)

## 💡 Exemplos de Uso

### Exemplo 1: Configuração Personalizada

```python
from TCC import SmartTrafficLight

# Criar instância
system = SmartTrafficLight(
    camera1_source=0,  # Primeira câmera USB
    camera2_source=1   # Segunda câmera USB
)

# Modificar parâmetros antes de iniciar (se necessário)
system.GREEN_TIME = 30  # Ajustar tempo base de verde
system.YELLOW_TIME = 4  # Ajustar tempo de amarelo

# Iniciar sistema
system.start()
```

### Exemplo 2: Usando Vídeos de Teste

```python
from TCC import SmartTrafficLight

# Usar arquivos de vídeo para testes
traffic_system = SmartTrafficLight(
    'test_video_rua_a.mp4',
    'test_video_rua_b.mp4'
)

traffic_system.start()
```

### Exemplo 3: Integração em Aplicação Maior

```python
from TCC import SmartTrafficLight
import time

class TrafficController:
    def __init__(self):
        self.system = SmartTrafficLight(0, 1)
        
    def run(self):
        try:
            self.system.start()
        except Exception as e:
            print(f"Erro: {e}")
        finally:
            self.system.stop()

# Uso
controller = TrafficController()
controller.run()
```

## 🔍 Troubleshooting

### Problema: Erros de compatibilidade ou código não funciona

**Sintomas**: Erros ao importar módulos, problemas com PyTorch, ou comportamento inesperado.

**Solução**: 
- ⚠️ **Verifique a versão do Python**: Este código foi desenvolvido para Python 3.8 até 3.11
- Versões acima de Python 3.11 (3.12, 3.13, etc.) **NÃO são compatíveis**
- Para verificar sua versão:
  ```bash
  python --version
  ```
- Se você tiver Python 3.12 ou superior, instale uma versão compatível (3.8, 3.9, 3.10 ou 3.11)
- Recomendado: Use Python 3.10 ou 3.11 para melhor compatibilidade

### Problema: Câmeras não são detectadas

**Solução**:
- Verifique se as câmeras estão conectadas
- Teste com `cv2.VideoCapture(0)` e `cv2.VideoCapture(1)` separadamente
- No Windows, verifique o Gerenciador de Dispositivos

### Problema: Modelo YOLOv5 não carrega

**Solução**:
```bash
# Reinstalar PyTorch e YOLOv5
pip uninstall torch torchvision
pip install torch torchvision
```

### Problema: Performance baixa (FPS baixo)

**Soluções**:
- Use GPU com CUDA se disponível
- Reduza a resolução das câmeras no código:
  ```python
  camera.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
  camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
  ```
- Use modelo YOLOv5 menor (yolov5n ao invés de yolov5s)

### Problema: Detecções falsas (falsos positivos)

**Soluções**:
- Ajuste o threshold de confiança:
  ```python
  self.yolo_model.conf = 0.5  # Aumentar para menos detecções
  ```
- Ajuste o threshold de estabilização:
  ```python
  # Em stabilize_detection(), alterar de 0.4 para 0.5 ou 0.6
  return (positive_detections / total_detections) >= 0.5
  ```

### Problema: Janelas não aparecem

**Solução**:
- Verifique se está usando interface gráfica (não funciona em SSH sem X11)
- No Windows, certifique-se de ter display conectado

### Problema: Erro ao usar vídeos

**Solução**:
- Verifique se os arquivos de vídeo existem
- Use codecs suportados (MP4, AVI com codec H.264)
- Verifique o caminho completo dos arquivos

## 📝 Notas Importantes

1. **Primeira Execução**: Na primeira execução, o YOLOv5 baixará o modelo automaticamente (~14MB). Isso pode levar alguns minutos.

2. **Calibração**: Os parâmetros do sistema (distâncias, espaçamentos) podem precisar ser ajustados conforme a configuração física da interseção.

3. **Threading**: O sistema usa duas threads separadas para detecção e controle, garantindo que o processamento de vídeo não interfira no controle dos semáforos.

4. **Normas MBST**: Os cálculos seguem as normas MBST Vol. V, mas podem ser ajustados conforme regulamentações locais.

## 📄 Licença

Este projeto foi desenvolvido para fins acadêmicos (TCC).

## 👤 Autor

Desenvolvido como parte do Trabalho de Conclusão de Curso (TCC).

---

**⚠️ Aviso**: Este sistema é para fins de pesquisa e demonstração. Para uso em produção, são necessários testes extensivos, certificações e aprovações regulatórias adequadas.
