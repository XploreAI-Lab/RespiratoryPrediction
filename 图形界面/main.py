import tkinter as tk
from tkinter import ttk
import os
import subprocess
# ------------------------------------------------------------
# AnimatedSidebarApp  (No‑slide version)
# ------------------------------------------------------------
# 侧边栏行为保持；主页面切换 **无动画**，即点击按钮立即显示对应页面。
# ------------------------------------------------------------

class AnimatedSidebarApp:
    def __init__(self, root):
        self.root = root
        self.root.title("呼吸频率预测")
        self.root.geometry("600x400")

        # ------------ 侧边栏参数 ------------
        self.sidebar_expanded = False
        self.sidebar_width = 100
        self.minimized_width = 40

        self.create_main_content()
        self.create_sidebar()
        self.show_page(self.pages[0])

    # ====================== 侧边栏 =====================
    def create_sidebar(self):
        init_w = self.sidebar_width if self.sidebar_expanded else self.minimized_width
        self.sidebar = tk.Frame(self.root, bg="#7f7f7f", width=init_w)
        self.sidebar.place(x=0, y=0, width=init_w, relheight=1)

        self.sidebar_interact = tk.Frame(self.sidebar, bg="#000000")
        self.sidebar_interact.pack(fill=tk.BOTH, expand=True)

        self.sidebar.bind("<Enter>", self.expand_sidebar)
        self.sidebar.bind("<Leave>", self.collapse_sidebar)

        self.nav_buttons = [
            ("🏠", "首页", self.show_home),
            ("🧪", "检测", self.show_settings),
            ("📚", "数据库", self.show_about),
            ("ℹ", "文档", self.show_docs)
        ]

        self.buttons = []
        style = ttk.Style()
        style.theme_use("clam")
        style.configure(
            "Icon.TButton",
            background="#000000",
            foreground="#00F0FF",
            borderwidth=0,
            font=("Segoe UI Symbol", 12),
            anchor="center",
        )
        style.map(
            "Icon.TButton",
            background=[("active", "#36c39d")],
            foreground=[("active", "#000000")],
        )

        for icon, text, cmd in self.nav_buttons:
            btn = ttk.Button(
                self.sidebar_interact,
                text=icon,
                style="Icon.TButton",
                command=cmd,
            )
            btn.pack(fill=tk.X, pady=2)
            self.buttons.append((btn, icon, text))
        self.update_buttons()

    def update_buttons(self):
        for btn, icon, text in self.buttons:
            if self.sidebar_expanded:
                btn.config(text=f"{icon} {text}", width=18)
            else:
                btn.config(text=icon, width=3)

    def expand_sidebar(self, _=None):
        if not self.sidebar_expanded:
            self.animate_sidebar(self.sidebar_width)
            self.sidebar_expanded = True
            self.update_buttons()

    def collapse_sidebar(self, _=None):
        if self.sidebar_expanded:
            self.animate_sidebar(self.minimized_width)
            self.sidebar_expanded = False
            self.update_buttons()

    def animate_sidebar(self, target_w):
        cur = self.sidebar.winfo_width()
        step = 10 if target_w > cur else -10

        def _anim():
            nonlocal cur
            if (step > 0 and cur < target_w) or (step < 0 and cur > target_w):
                cur += step
                self.sidebar.place_configure(width=cur)
                self.root.after(10, _anim)
            else:
                self.sidebar.place_configure(width=target_w)

        _anim()

    # ====================== 主内容 =====================
    def create_main_content(self):
        self.main_canvas = tk.Canvas(self.root, bg="#f0f0f0", highlightthickness=0)
        self.main_canvas.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.pages = [
            self.create_page("🏠 首页", "#FFB74D"),
            self.create_page("🧪 检测", "#4DB6AC"),
            self.create_page("ℹ 数据", "#7986CB"),
            self.create_page("📚 文档", "#A1887F"),
        ]
        # 先隐藏所有页
        for pg in self.pages:
            pg.place_forget()

    def create_page(self, text, color):
        f = tk.Frame(self.main_canvas, bg=color)
        tk.Label(f, text=text, font=("微软雅黑", 24), bg=color).pack(pady=50)
        return f

    # ---------------------- 页面切换（无动画） ----------------------
    def show_page(self, page):
        current = next((p for p in self.pages if p.winfo_viewable()), None)
        if current is page:
            return
        if current:
            current.place_forget()
        page.place(x=0, y=0, relwidth=1, relheight=1)

    # 导航回调
    def show_home(self):
        self.show_page(self.pages[0])

    def show_settings(self):
        self.show_page(self.pages[1])

        # 在检测页面上添加运行按钮
        run_button = tk.Button(
            self.pages[1],
            text="Start",
            command=self.run,
            bg="#90EE90",  # 浅绿色背景
            activebackground="#7CCD7C",  # 点击时的背景色
            fg="black",  # 文字颜色
            font=("Arial", 12, "bold"),  # 字体设置
            width=6,  # 按钮宽度（字符单位）
            height=1  # 按钮高度（行数单位）
        )

        # 使用place布局管理器精确定位
        run_button.place(relx=0.5,  # 水平居中 (50%)
                         rely=0.4,  # 垂直位置 (60%高度处)
                         anchor="center")  # 以按钮中心为锚点
        # run_button.pack(pady=10)

    def run(self):
        script_path = os.path.join(os.path.dirname(__file__), "show_signal.py")
        subprocess.Popen(["python", script_path], shell=True)
    def show_about(self):
        self.show_page(self.pages[2])

    def show_docs(self):
        self.show_page(self.pages[3])


if __name__ == "__main__":
    root = tk.Tk()
    app = AnimatedSidebarApp(root)
    root.mainloop()