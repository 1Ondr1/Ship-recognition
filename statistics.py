import os
import pandas as pd
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import torch
from torchvision import transforms
from PIL import Image
from train import ResNet18

class StatisticsApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Ship Classifier Statistics")
        self.root.geometry("600x200")
        self.root.resizable(False, False)
        
        self.model = None
        self.image_folder = ""
        self.csv_path = ""
        
        self.create_widgets()

    def create_widgets(self):
        frame = ttk.Frame(self.root)
        frame.pack(pady=20)

        ttk.Label(frame, text="Model Path:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.model_path_entry = ttk.Entry(frame, width=50)
        self.model_path_entry.grid(row=0, column=1, padx=5, pady=5)
        ttk.Button(frame, text="Browse", command=self.load_model).grid(row=0, column=2, padx=5, pady=5)

        ttk.Label(frame, text="Image Folder:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        self.image_folder_entry = ttk.Entry(frame, width=50)
        self.image_folder_entry.grid(row=1, column=1, padx=5, pady=5)
        ttk.Button(frame, text="Browse", command=self.choose_image_folder).grid(row=1, column=2, padx=5, pady=5)

        ttk.Label(frame, text="CSV Path:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=5)
        self.csv_path_entry = ttk.Entry(frame, width=50)
        self.csv_path_entry.grid(row=2, column=1, padx=5, pady=5)
        ttk.Button(frame, text="Browse", command=self.choose_csv_file).grid(row=2, column=2, padx=5, pady=5)

        ttk.Button(self.root, text="Calculate Statistics", command=self.calculate_statistics).pack(pady=10)

    def load_model(self):
        model_path = filedialog.askopenfilename(initialdir=os.getcwd(), title="Select Model", filetypes=[("PyTorch Model", "*.pth")])
        if model_path:
            self.model_path_entry.delete(0, tk.END)
            self.model_path_entry.insert(0, model_path)
            self.model = ResNet18(num_classes=5)
            self.model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
            self.model.eval()
            messagebox.showinfo("Info", "Model loaded successfully!")

    def choose_image_folder(self):
        folder_path = filedialog.askdirectory(initialdir=os.getcwd(), title="Select Image Folder")
        if folder_path:
            self.image_folder_entry.delete(0, tk.END)
            self.image_folder_entry.insert(0, folder_path)
            self.image_folder = folder_path

    def choose_csv_file(self):
        csv_path = filedialog.askopenfilename(initialdir=os.getcwd(), title="Select CSV File", filetypes=[("CSV Files", "*.csv")])
        if csv_path:
            self.csv_path_entry.delete(0, tk.END)
            self.csv_path_entry.insert(0, csv_path)
            self.csv_path = csv_path

    def calculate_statistics(self):
        if not self.model or not self.image_folder or not self.csv_path:
            messagebox.showerror("Error", "Please make sure to load model, choose image folder, and CSV file.")
            return

        df = pd.read_csv(self.csv_path)
        correct = 0
        total = 0
        transform = transforms.Compose([
            transforms.Resize((100, 100)),
            transforms.ToTensor(),
        ])
        
        for idx, row in df.iterrows():
            img_name, true_label = row[0], row[1]
            img_path = os.path.join(self.image_folder, img_name)
            if os.path.exists(img_path):
                image = Image.open(img_path).convert("RGB")
                image = transform(image).unsqueeze(0)
                
                with torch.no_grad():
                    output = self.model(image)
                predicted_label = torch.argmax(output).item() + 1  # Adjusting for 1-based index

                if predicted_label == true_label:
                    correct += 1
                total += 1

        accuracy = (correct / total) * 100 if total else 0
        result_text = f"Correct Predictions: {correct}\nTotal Predictions: {total}\nAccuracy: {accuracy:.2f}%"
        messagebox.showinfo("Statistics", result_text)

if __name__ == "__main__":
    root = tk.Tk()
    app = StatisticsApp(root)
    root.mainloop()
