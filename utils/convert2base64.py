import base64
import os


def convert_images_to_base64_files(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        file_path = os.path.join(input_folder, filename)
        output_file_path = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}.txt")

        try:
            with open(file_path, "rb") as image_file:
                base64_string = base64.b64encode(image_file.read()).decode('utf-8')

            with open(output_file_path, "w") as output_file:
                output_file.write(base64_string)

            print(f"Converted {filename} to base64 and saved to {output_file_path}.")
        except Exception as e:
            print(f"Could not process {filename}: {e}")

input_folder = "../images"
output_folder = "../base64"
os.makedirs(output_folder, exist_ok=True)
convert_images_to_base64_files(input_folder, output_folder)