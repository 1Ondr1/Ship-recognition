import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import torch
from torchvision import transforms
from PIL import Image, ImageTk
from train import ResNet18 

class ShipClassifierApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Ship Classifier")
        self.root.geometry("800x600")
        self.root.resizable(False, False)
        
        self.model = None
        self.image = None

        self.style = ttk.Style()
        self.style.configure('TButton', font=('Helvetica', 12), padding=10)
        self.style.configure('TLabel', font=('Helvetica', 12), padding=10)
        self.style.configure('TEntry', font=('Helvetica', 12), padding=10)

        self.frame = ttk.Frame(root)
        self.frame.pack(pady=20)

        self.model_path_label = ttk.Label(self.frame, text="Model Path:")
        self.model_path_label.grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.model_path_entry = ttk.Entry(self.frame, width=50)
        self.model_path_entry.grid(row=0, column=1, padx=5, pady=5)
        self.load_model_button = ttk.Button(self.frame, text="Browse", command=self.load_model)
        self.load_model_button.grid(row=0, column=2, padx=5, pady=5)

        self.image_path_label = ttk.Label(self.frame, text="Image Path:")
        self.image_path_label.grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        self.image_path_entry = ttk.Entry(self.frame, width=50)
        self.image_path_entry.grid(row=1, column=1, padx=5, pady=5)
        self.load_image_button = ttk.Button(self.frame, text="Browse", command=self.load_image, state=tk.DISABLED)
        self.load_image_button.grid(row=1, column=2, padx=5, pady=5)

        self.image_label = ttk.Label(self.frame, text="No image loaded", compound=tk.CENTER)
        self.image_label.grid(row=2, column=0, columnspan=3, padx=5, pady=20)

        self.result_label = ttk.Label(self.frame, text="", anchor="center")
        self.result_label.grid(row=3, column=0, columnspan=3, padx=5, pady=20)

    def load_model(self):
        model_path = filedialog.askopenfilename(initialdir=os.getcwd(), title="Select Model", filetypes=[("PyTorch Model", "*.pth")])
        if model_path:
            self.model_path_entry.delete(0, tk.END)
            self.model_path_entry.insert(0, model_path)
            self.model = ResNet18(num_classes=5)
            self.model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
            self.model.eval()
            self.load_image_button.config(state=tk.NORMAL)
            messagebox.showinfo("Info", "Model loaded successfully!")

    def load_image(self):
        image_path = filedialog.askopenfilename(initialdir=os.getcwd(), title="Select Image", filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")])
        if image_path:
            self.image_path_entry.delete(0, tk.END)
            self.image_path_entry.insert(0, image_path)
            image = Image.open(image_path).convert("RGB")
            transform = transforms.Compose([
                transforms.Resize((100, 100)),
                transforms.ToTensor(),
            ])
            self.image = transform(image).unsqueeze(0)

            image_display = ImageTk.PhotoImage(image.resize((200, 200)))
            self.image_label.config(image=image_display, text="")
            self.image_label.image = image_display

            self.classify_image()

    def classify_image(self):
        if self.model is not None and self.image is not None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(device)
            self.image = self.image.to(device)

            with torch.no_grad():
                outputs = self.model(self.image)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            predicted_class = torch.argmax(probabilities).item()

            class_names = ["Cargo", "Military", "Carrier", "Cruise", "Tankers"]
            result_text = f"Predicted class: {class_names[predicted_class]}\n\nClass probabilities:\n"
            for i, prob in enumerate(probabilities):
                result_text += f"{class_names[i]}: {prob:.4f}\n"

            self.result_label.config(text=result_text)

if __name__ == "__main__":
    root = tk.Tk()
    app = ShipClassifierApp(root)
    root.mainloop()
