import torch

MODEL_PATH = 'model.pth'  # Убедитесь, что имя файла верное

try:
    # Загружаем файл
    state_dict = torch.load(MODEL_PATH, map_location='cpu')
    
    # Если файл сохранен целиком (не словарь)
    if hasattr(state_dict, 'state_dict'):
        print("✅ Файл содержит ПОЛНУЮ модель (не только веса).")
        print("Вам повезло! Используйте: model = torch.load('model.pth')")
        print("Архитектура:")
        print(state_dict)
    
    # Если это словарь весов (как обычно)
    elif isinstance(state_dict, dict):
        print("ℹ️ Файл содержит словарь весов (state_dict).")
        keys = list(state_dict.keys())
        print(f"Всего ключей (слоев): {len(keys)}")
        print("\nПервые 10 ключей (по ним можно понять архитектуру):")
        for k in keys[:10]:
            print(f" - {k}")
            
        print("\nПоследние ключи:")
        for k in keys[-5:]:
            print(f" - {k}")
            
    else:
        print("⚠️ Непонятный формат содержимого.")

except Exception as e:
    print(f"Ошибка чтения файла: {e}")