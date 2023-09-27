import os
import cv2
import tkinter as tk
from tkinter import Tk
from cv2 import dnn_superres
from tkinter import messagebox
from tkinter.filedialog import askopenfilename

def EDSR(): #Enhanced Deep Super-Resolution (EDSR)
    # Menginisialisasi objek resolusi super
    sr = dnn_superres.DnnSuperResImpl_create()

    # Baca modelnya
    path = 'data/EDSR_x4.pb'
    sr.readModel(path)

    # Mengatur model dan skala
    sr.setModel('edsr', 4)

    sr.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    sr.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

    # membuat jendela root Tkinter
    root = Tk()
    root.withdraw()

    # Meminta pengguna untuk memilih file gambar
    file_path = askopenfilename(filetypes=[('Image Files', ('*.png', '*.jpg', '*.jpeg', '*.bmp'))])

    # Memeriksa apakah file dipilih
    if file_path:
        # Muat gambar
        image = cv2.imread(file_path)

        # Upsample gambar
        upscaled = sr.upsample(image)

        # Tentukan folder keluaran
        output_folder = '.output'

        # Buat folder keluaran jika tidak ada
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        # Simpan gambar yang ditingkatkan
        upscaled_path = os.path.join(output_folder, 'upscaled_test.png')
        
        cv2.imwrite(upscaled_path, upscaled)

        messagebox.showwarning("Enhance", "Enhance Images succeed")
    else:
        messagebox.showwarning("file selected", "No image file selected !")

    # Tutup jendela root Tkinter
    root.destroy()
    
def main():
    window = tk.Tk()
    window.title("EDSR") 
    window.geometry("300x451")  # Atur ukuran tampilan menjadi 300x451 piksel
    window.resizable(False, False)  # Nonaktifkan perubahan ukuran jendela

    background_image = tk.PhotoImage(file="data/Gui.png") # Mengatur latar belakang
    background_label = tk.Label(window, image=background_image)
    background_label.place(x=0, y=0, relwidth=1, relheight=1)

    button_width = 15  # Tentukan lebar tombol

    button = tk.Button(window, text="Enhance", command=EDSR, width=button_width)
    button.place(x=95, y=350)

    window.mainloop()

if __name__ == "__main__":
    main()
