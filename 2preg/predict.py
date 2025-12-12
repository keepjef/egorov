import sys
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image

# ================= НАСТРОЙКИ =================
MODEL_PATH = 'model.pth'   # Имя вашего файла
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
# =============================================

# -----------------------------------------------------------
# 1. СЮДА НУЖНО ВСТАВИТЬ КОД КЛАССА ВАШЕЙ МОДЕЛИ
# Если у вас есть файл model.py, скопируйте класс оттуда.
# Ниже приведен пример стандартного U-Net.
# -----------------------------------------------------------

class UNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=3):
        super(UNet, self).__init__()
        # Временная заглушка, чтобы код запустился.
        # ВАМ НУЖЕН НАСТОЯЩИЙ КОД ВАШЕЙ МОДЕЛИ!
        # Если ключи в файле не совпадут с этим кодом, выпадет ошибка.
        self.inc = nn.Conv2d(n_channels, 64, kernel_size=3, padding=1)
        # ... (здесь должна быть вся архитектура) ... 

# Если вы не знаете архитектуру, этот код упадет на шаге load_state_dict.
# Но структуру исправления ошибки мы сейчас наладим:

def run_inference(image_path, output_path):
    print(f"Device: {DEVICE}")

    # --- ИСПРАВЛЕНИЕ ВАШЕЙ ОШИБКИ ЗДЕСЬ ---
    
    # Шаг 1. Создаем "пустую" модель (Скелет)
    # Замените UNet() на название вашего класса, если оно другое
    try:
        model = UNet(n_channels=3, n_classes=3) 
    except NameError:
        print("ОШИБКА: Вы не вставили класс модели в скрипт!")
        return

    # Шаг 2. Загружаем веса из файла (Словарь/OrderedDict)
    print(f"Загрузка весов из {MODEL_PATH}...")
    weights = torch.load(MODEL_PATH, map_location=DEVICE)

    # Шаг 3. Загружаем веса в модель
    try:
        model.load_state_dict(weights)
    except RuntimeError as e:
        print("\n!!! КРИТИЧЕСКАЯ ОШИБКА НЕСОВПАДЕНИЯ АРХИТЕКТУРЫ !!!")
        print("Код класса UNet в этом скрипте НЕ СОВПАДАЕТ с тем, который использовался при обучении.")
        print("В файле .pth ключи одни, а в коде скрипта — другие.")
        print("РЕШЕНИЕ: Найдите оригинальный код (файл .py), которым создавали этот .pth, и скопируйте класс модели сюда.")
        print(f"\nДетали ошибки: {e}")
        return

    # Шаг 4. Теперь у модели есть метод .to()
    model = model.to(DEVICE)
    model.eval()
    # --------------------------------------

    # Дальше всё как обычно: обработка картинки
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    try:
        img = Image.open(image_path).convert('RGB')
    except FileNotFoundError:
        print(f"Не найдена картинка: {image_path}")
        return

    input_tensor = transform(img).unsqueeze(0).to(DEVICE)

    print("Генерация...")
    with torch.no_grad():
        output_tensor = model(input_tensor)

    output_tensor = output_tensor.squeeze(0).cpu()
    to_pil = transforms.ToPILImage()
    res_img = to_pil(output_tensor)
    
    res_img.save(output_path)
    print(f"Готово -> {output_path}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Run: python predict.py input.jpg output.png")
    else:
        run_inference(sys.argv[1], "result.png")