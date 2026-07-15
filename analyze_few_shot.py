import os
import pandas as pd

def extract_accuracy_with_custom_rules(root_dir, target_indices, output_csv="summary_accuracy.csv"):
    results = []
    
    if not os.path.exists(root_dir):
        print(f"Ошибка: Директория '{root_dir}' не найдена.")
        return
        
    for item in sorted(os.listdir(root_dir)):
        subfolder_path = os.path.join(root_dir, item)
        
        if os.path.isdir(subfolder_path):
            file_path = os.path.join(subfolder_path, "ds_code.csv")
            
            if os.path.exists(file_path):
                try:
                    # Читаем CSV
                    df = pd.read_csv(file_path)
                    
                    # 1. Базовая статистика по всему файлу
                    total_examples = len(df)
                    correct_examples = int((df['accuracy'] == 1).sum())
                    calculated_accuracy = correct_examples / total_examples if total_examples > 0 else 0.0
                    
                    # 2. Логика фильтрации для обновленной метрики (updated_accuracy)
                    if "auto_cot" in item:
                        # Исключаем индексы 125, 155, 136
                        exclude_indices = [125, 155, 136]
                    else:
                        # Исключаем индексы 0, 82, 163
                        exclude_indices = [0, 82, 163]
                    
                    # Дропаем строки по их порядковому индексу
                    valid_exclude_indices = [idx for idx in exclude_indices if idx < len(df)]
                    df_updated = df.drop(index=valid_exclude_indices)
                    
                    # Считаем обновленную метрику
                    updated_total = len(df_updated)
                    updated_correct = int((df_updated['accuracy'] == 1).sum())
                    updated_accuracy = updated_correct / updated_total if updated_total > 0 else 0.0
                    
                    # 3. Собираем данные по конкретным few-shot индексам для вывода в таблицу
                    row_data = {}
                    for idx in target_indices:
                        if idx < len(df):
                            t_id = df.iloc[idx]['task_id']
                            acc = df.iloc[idx]['accuracy']
                            row_data[idx] = (t_id, acc)
                        else:
                            row_data[idx] = (f"index_{idx}_out_of_bounds", None)
                    
                    # Формируем строку для результирующего CSV
                    res_row = {
                        "folder_name": item,
                        "total_examples": total_examples,
                        "correct_examples": correct_examples,
                        "calculated_accuracy": calculated_accuracy,
                        "updated_accuracy": updated_accuracy
                    }
                    
                    # Добавляем колонки для конкретных задач
                    for idx in target_indices:
                        t_id, acc = row_data[idx]
                        column_name = f"task_{idx} ({t_id})" if not isinstance(t_id, str) or not t_id.startswith("index") else f"task_{idx}"
                        res_row[column_name] = acc
                        
                    results.append(res_row)
                    print(f"Успешно обработано: {item}")
                except Exception as e:
                    print(f"Ошибка при обработке файла {file_path}: {e}")
            else:
                print(f"Файл ds_code.csv не найден в папке: {item}")

    if results:
        output_df = pd.DataFrame(results)
        
        # Округляем расчетные метрики до 4 знаков после запятой
        output_df["calculated_accuracy"] = output_df["calculated_accuracy"].round(4)
        output_df["updated_accuracy"] = output_df["updated_accuracy"].round(4)
        
        # Определяем порядок колонок, чтобы вся статистика была в начале
        base_cols = ["folder_name", "total_examples", "correct_examples", "calculated_accuracy", "updated_accuracy"]
        task_cols = [c for c in output_df.columns if c not in base_cols]
        output_df = output_df[base_cols + task_cols]
        
        output_df.to_csv(output_csv, index=False, encoding='utf-8-sig')
        print(f"\nГотово! Результаты сохранены в файл: {output_csv}")
    else:
        print("\nДанные для сохранения не найдены.")

if __name__ == "__main__":
    ROOT_DIRECTORY = "results_codemmlu"
    TARGET_INDICES = [0, 82, 163, 125, 155, 136]
    OUTPUT_FILE = "summary_accuracy.csv"
    
    extract_accuracy_with_custom_rules(ROOT_DIRECTORY, TARGET_INDICES, OUTPUT_FILE)