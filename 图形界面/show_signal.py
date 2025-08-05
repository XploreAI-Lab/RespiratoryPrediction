import tkinter as tk
from tkinter import ttk
import serial
import struct
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import threading
import queue
import time


class ECGPPGMonitor:
    def __init__(self, master):
        self.master = master
        master.title("ECG & PPG Monitor")
        master.configure(bg='black')

        # ECG协议参数（保持不变）
        self.CES_CMDIF_PKT_START_1 = 0x0A
        self.CES_CMDIF_PKT_START_2 = 0xFA
        self.CES_CMDIF_PKT_STOP = 0x0B
        self.CES_CMDIF_IND_LEN = 2
        self.CES_CMDIF_IND_LEN_MSB = 3
        self.CES_CMDIF_IND_PKTTYPE = 4
        self.CES_CMDIF_PKT_OVERHEAD = 5

        # 状态变量（保持不变）
        self.pc_rx_state = 0
        self.CES_Pkt_Len = 0
        self.CES_Pkt_Pos_Counter = 0
        self.CES_Data_Counter = 0
        self.CES_Pkt_PktType = 0
        self.CES_Pkt_Data_Counter = bytearray(1000)

        # 数据缓冲区（保持不变）
        self.window_size = 500
        self.xdata = np.arange(self.window_size)
        self.ecg_data = np.zeros(self.window_size)
        self.ppg_data = np.zeros(self.window_size)
        self.array_index = 0

        # 创建GUI
        self.create_widgets()

        # 串口相关（保持不变）
        self.ecg_ser = None
        self.ppg_ser = None
        self.data_queue = queue.Queue()
        self.is_running = False

    def create_widgets(self):
        # 设置全局样式
        style = ttk.Style()
        style.theme_use('alt')
        style.configure('.', background='black', foreground='white')
        style.configure('TFrame', background='black')
        style.configure('TLabel', background='black', foreground='white')
        style.configure('TButton',
                        background='#333333',
                        foreground='white',
                        bordercolor='#444444',
                        borderwidth=1)
        style.map('TButton',
                  background=[('active', '#444444'), ('disabled', '#222222')])
        style.configure('TCombobox',
                        fieldbackground='#333333',
                        background='#333333',
                        foreground='white')
        style.map('TCombobox',
                  fieldbackground=[('readonly', '#333333')],
                  background=[('readonly', '#333333')])

        # 控制面板
        control_frame = ttk.Frame(self.master, padding=10)
        control_frame.pack(side=tk.TOP, fill=tk.X)

        # ECG控件
        ttk.Label(control_frame, text="ECG Port:").grid(row=0, column=0, padx=5)
        self.ecg_port_var = tk.StringVar()
        self.ecg_port_combo = ttk.Combobox(control_frame, textvariable=self.ecg_port_var, width=15)
        self.ecg_port_combo.grid(row=0, column=1, padx=5)

        ttk.Label(control_frame, text="ECG Baud:").grid(row=0, column=2, padx=5)
        self.ecg_baud_var = tk.StringVar(value="57600")
        self.ecg_baud_combo = ttk.Combobox(control_frame,
                                           values=["9600", "57600", "115200"],
                                           textvariable=self.ecg_baud_var,
                                           width=8)
        self.ecg_baud_combo.grid(row=0, column=3, padx=5)

        # PPG控件
        ttk.Label(control_frame, text="PPG Port:").grid(row=1, column=0, padx=5)
        self.ppg_port_var = tk.StringVar()
        self.ppg_port_combo = ttk.Combobox(control_frame, textvariable=self.ppg_port_var, width=15)
        self.ppg_port_combo.grid(row=1, column=1, padx=5)

        ttk.Label(control_frame, text="PPG Baud:").grid(row=1, column=2, padx=5)
        self.ppg_baud_var = tk.StringVar(value="115200")
        self.ppg_baud_combo = ttk.Combobox(control_frame,
                                           values=["9600", "57600", "115200"],
                                           textvariable=self.ppg_baud_var,
                                           width=8)
        self.ppg_baud_combo.grid(row=1, column=3, padx=5)

        # 控制按钮
        self.start_btn = ttk.Button(control_frame,
                                    text="Start",
                                    width=8,
                                    command=self.toggle_connection)
        self.start_btn.grid(row=0, column=4, padx=10)

        self.refresh_btn = ttk.Button(control_frame,
                                    text="Refresh",
                                    width=8,
                                    command=self.refresh_line)
        self.refresh_btn.grid(row=1, column=4, padx=10)
        self.refresh_ports()

        # 波形显示区域
        fig = Figure(figsize=(10, 1), dpi=100)
        fig.subplots_adjust(hspace=0.6)
        fig.patch.set_facecolor('black')  # 画布背景

        # ECG子图设置
        self.ax1 = fig.add_subplot(211)
        self.ax1.set_title("ECG Signal", color='#00FF00', fontsize=12, pad=20)
        self.ax1.set_facecolor('black')
        self.ax1.tick_params(axis='both', colors='white')
        for spine in self.ax1.spines.values():
            spine.set_color('white')
        self.ax1.set_ylim(-20, 20)
        self.ecg_line, = self.ax1.plot(self.xdata, self.ecg_data, color='#00FF00', linewidth=1)

        # PPG子图设置
        self.ax2 = fig.add_subplot(212)
        self.ax2.set_title("PPG Signal", color='cyan', fontsize=12, pad=20)
        self.ax2.set_facecolor('black')
        self.ax2.tick_params(axis='both', colors='white')
        for spine in self.ax2.spines.values():
            spine.set_color('white')
        self.ax2.set_ylim(0, 1200)
        self.ppg_line, = self.ax2.plot(self.xdata, self.ppg_data, color='cyan', linewidth=1)

        # 嵌入画布
        self.canvas = FigureCanvasTkAgg(fig, master=self.master)
        self.canvas.get_tk_widget().configure(bg='black')  # 画布控件背景
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # 状态栏
        status_frame = ttk.Frame(self.master, padding=10)
        status_frame.pack(side=tk.BOTTOM, fill=tk.X)
        self.hr_label = ttk.Label(status_frame,
                                  text="Heart Rate: -- bpm",
                                  font=('Arial', 12, 'bold'),
                                  foreground='#F0B020')
        self.hr_label.pack(side=tk.LEFT, padx=20)

    def refresh_ports(self):
        ports = [f"COM{i + 1}" for i in range(256)]
        self.ecg_port_combo['values'] = ports
        self.ppg_port_combo['values'] = ports

    def toggle_connection(self):
        if self.is_running:
            self.stop()
        else:
            self.start()

    def refresh_line(self):
        self.ecg_data = np.zeros(self.window_size)
        self.ppg_data = np.zeros(self.window_size)
        self.ecg_line.set_ydata(self.ecg_data)
        self.ppg_line.set_ydata(self.ppg_data)
        self.canvas.draw_idle()

    def start(self):
        try:
            self.ecg_ser = serial.Serial(self.ecg_port_var.get(), int(self.ecg_baud_var.get()), timeout=0.1)
            self.ppg_ser = serial.Serial(self.ppg_port_var.get(), int(self.ppg_baud_var.get()), timeout=0.1)
            self.is_running = True
            self.start_btn.config(text="Stop")
            threading.Thread(target=self.read_ecg_serial, daemon=True).start()
            threading.Thread(target=self.read_ppg_serial, daemon=True).start()
            self.master.after(10, self.update_plot)
        except Exception as e:
            print(f"Error: {e}")

    def stop(self):
        self.is_running = False
        if self.ecg_ser and self.ecg_ser.is_open:
            self.ecg_ser.close()
        if self.ppg_ser and self.ppg_ser.is_open:
            self.ppg_ser.close()
        self.start_btn.config(text="Start")

    def read_ecg_serial(self):
        while self.is_running and self.ecg_ser.is_open:
            data = self.ecg_ser.read(1)
            if data:
                self.process_ecg_data(data[0])

    def read_ppg_serial(self):
        while self.is_running and self.ppg_ser.is_open:
            if self.ppg_ser.in_waiting:
                line = self.ppg_ser.readline().decode('utf-8').strip()
                if line[0] == 'S':
                    try:
                        val = float(line[1:])
                        self.ppg_data[self.array_index] = val
                    except:
                        pass

    def process_ecg_data(self, rxch):
        if self.pc_rx_state == 0:  # Init
            if rxch == self.CES_CMDIF_PKT_START_1:
                self.pc_rx_state = 1
        elif self.pc_rx_state == 1:  # SOF1 Found
            if rxch == self.CES_CMDIF_PKT_START_2:
                self.pc_rx_state = 2
            else:
                self.pc_rx_state = 0
        elif self.pc_rx_state == 2:  # SOF2 Found
            self.CES_Pkt_Len = rxch
            self.CES_Pkt_Pos_Counter = self.CES_CMDIF_IND_LEN
            self.CES_Data_Counter = 0
            self.pc_rx_state = 3
        elif self.pc_rx_state == 3:  # Len Found
            self.CES_Pkt_Pos_Counter += 1
            if self.CES_Pkt_Pos_Counter < self.CES_CMDIF_PKT_OVERHEAD:
                if self.CES_Pkt_Pos_Counter == self.CES_CMDIF_IND_LEN_MSB:
                    self.CES_Pkt_Len = (rxch << 8) | self.CES_Pkt_Len
                elif self.CES_Pkt_Pos_Counter == self.CES_CMDIF_IND_PKTTYPE:
                    self.CES_Pkt_PktType = rxch
            else:
                if self.CES_Pkt_PktType == 2:
                    if self.CES_Data_Counter < len(self.CES_Pkt_Data_Counter):
                        self.CES_Pkt_Data_Counter[self.CES_Data_Counter] = rxch
                        self.CES_Data_Counter += 1
                if self.CES_Pkt_Pos_Counter >= self.CES_CMDIF_PKT_OVERHEAD + self.CES_Pkt_Len + 1:
                    if rxch == self.CES_CMDIF_PKT_STOP:
                        self.handle_ads1292r_data()
                    self.pc_rx_state = 0

    def update_plot(self):
        while not self.data_queue.empty():
            ecg, resp, hr, rr = self.data_queue.get()
            self.hr_label.config(text=f"Heart Rate: {hr} bpm")

        #滚动数据
        self.ecg_line.set_ydata(np.roll(self.ecg_data, -self.array_index))
        self.ppg_line.set_ydata(np.roll(self.ppg_data, -self.array_index))
        self.canvas.draw_idle()

        if self.is_running:
            self.master.after(50, self.update_plot)

    def handle_ads1292r_data(self):
        # 解析ADS1292R数据包
        ecg_raw = struct.unpack('<h', self.CES_Pkt_Data_Counter[0:2])[0]
        resp_raw = struct.unpack('<h', self.CES_Pkt_Data_Counter[2:4])[0]
        rr = struct.unpack('<h', self.CES_Pkt_Data_Counter[4:6])[0]
        hr = struct.unpack('<h', self.CES_Pkt_Data_Counter[6:8])[0]

        # 更新数据缓冲区
        self.ecg_data[self.array_index] = ecg_raw
        self.array_index = (self.array_index + 1) % self.window_size

        # 更新界面显示
        self.data_queue.put((ecg_raw, resp_raw, hr, rr))

if __name__ == "__main__":
    root = tk.Tk()
    app = ECGPPGMonitor(root)
    root.geometry("900x500")
    root.mainloop()
