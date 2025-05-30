import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from train import train_ship_classifier

class ShipClassifierApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Ship Classifier")
        self.root.geometry("600x250")
        self.root.resizable(False, False)
        
        self.model_path = tk.StringVar()
        self.sample_frac = tk.StringVar(value="1.0")
        self.process_images = tk.BooleanVar(value=True)

        self.create_widgets()

    def create_widgets(self):
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        model_frame = ttk.LabelFrame(main_frame, text="Model Details", padding="10")
        model_frame.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

        ttk.Label(model_frame, text="Model Path:").grid(row=0, column=0, padx=5, pady=5, sticky=tk.E)
        ttk.Entry(model_frame, textvariable=self.model_path, width=50).grid(row=0, column=1, padx=5, pady=5)
        ttk.Button(model_frame, text="Browse", command=self.choose_model_file).grid(row=0, column=2, padx=5, pady=5)

        ttk.Label(model_frame, text="Sample Fraction:").grid(row=1, column=0, padx=5, pady=5, sticky=tk.E)
        ttk.Entry(model_frame, textvariable=self.sample_frac).grid(row=1, column=1, padx=5, pady=5)

        ttk.Checkbutton(model_frame, text="Process Images", variable=self.process_images).grid(row=2, column=0, columnspan=3, padx=5, pady=5)

        ttk.Button(main_frame, text="Train Model", command=self.train_model).pack(pady=10)

    def choose_model_file(self):
        file_path = filedialog.asksaveasfilename(initialdir=os.getcwd(), defaultextension=".pth", filetypes=[("PyTorch Model", "*.pth")])
        if file_path:
            self.model_path.set(file_path)

    def train_model(self):
        model_path = self.model_path.get()
        sample_frac = self.sample_frac.get()
        process_images = self.process_images.get()

        try:
            sample_frac = float(sample_frac)
            if not 0 < sample_frac <= 1:
                raise ValueError
        except ValueError:
            messagebox.showerror("Invalid Input", "Sample fraction must be a number between 0 and 1.")
            return

        if not model_path:
            messagebox.showerror("Error", "Please choose a model file.")
            return

        train_ship_classifier(model_path, sample_frac, process_images)
        messagebox.showinfo("Training Complete", "Model training completed successfully.")

if __name__ == "__main__":
    root = tk.Tk()
    app = ShipClassifierApp(root)
    root.mainloop()
