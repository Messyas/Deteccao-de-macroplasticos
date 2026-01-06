import os
import zipfile
import shutil
import tempfile 

def extract_nested_images_robust(source_dir):
    output_dir = os.path.join(source_dir, 'all_images')
    os.makedirs(output_dir, exist_ok=True)
    print(f"Diretório de origem: {source_dir}")
    print(f"Diretório de saída: {output_dir}")
    print("-" * 30)

    image_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff')
    found_zips = False
    extracted_count = 0

    for item in os.listdir(source_dir):
        item_path = os.path.join(source_dir, item)

        if os.path.isfile(item_path) and item.lower().endswith('.zip'):
            found_zips = True
            print(f"\nProcessando arquivo ZIP: {item}")

            with tempfile.TemporaryDirectory() as temp_extract_path:
                try:
                    with zipfile.ZipFile(item_path, 'r') as zip_ref:
                        zip_ref.extractall(temp_extract_path)
                    print(f"diretório extraído")

                    for root, dirs, files in os.walk(temp_extract_path):
                        dirs[:] = [d for d in dirs if not d.startswith('.') and not d.startswith('__')]
                        for file_name in files:
                            if file_name.lower().endswith(image_extensions) and not file_name.startswith('._'):
                                source_file_path = os.path.join(root, file_name)
                                target_file_path = os.path.join(output_dir, file_name)

                                print(f"Encontrada imagem: {os.path.relpath(source_file_path, temp_extract_path)}")

                                count = 1
                                base, ext = os.path.splitext(file_name)
                                while os.path.exists(target_file_path):
                                    target_file_path = os.path.join(output_dir, f"{base}_{count}{ext}")
                                    count += 1
                                shutil.copy2(source_file_path, target_file_path)
                                print(f"Copiada para: {os.path.basename(target_file_path)}")
                                extracted_count += 1

                except zipfile.BadZipFile:
                    print(f"O arquivo {item} não é um ZIP válido ou está corrompido.")
                except Exception as e:
                    print(f"ERRO inesperado ao processar {item}: {e}")

    print("-" * 30)
    if not found_zips:
        print("Nenhum arquivo .zip foi encontrado no diretório de origem.")
    else:
        print(f"Processamento concluído! {extracted_count} imagens foram copiadas para '{output_dir}'.")

SOURCE_PATH = '/home/messyas/Downloads/datasetMessyas'
extract_nested_images_robust(SOURCE_PATH)
