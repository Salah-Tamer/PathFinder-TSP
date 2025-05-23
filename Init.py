import tkinter as tk
from tkinter import ttk, messagebox, font
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import random
from matplotlib.animation import FuncAnimation
import matplotlib.colors as mcolors
from PIL import Image, ImageTk, ImageDraw, ImageFilter, ImageFont
import os
import colorsys
from TSPSolver import TSPSolver
import sys
import time
import matplotlib.gridspec as gridspec
from animation.NearestNeighbor import DEFAULT_STYLE

# Add the animation directory to the path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'animation'))
from animation.NearestNeighbor import visualize_nearest_neighbor
from animation.BruteForce import visualize_brute_force
import matplotlib.patches as mpatches

class ModernButton(tk.Canvas):
    def __init__(self, master=None, text="Button", command=None, width=120, height=40, 
                 corner_radius=10, bg_color="#4a86e8", fg_color="white", 
                 hover_color="#3a76d8", **kwargs):
        super().__init__(master, width=width, height=height,
                         highlightthickness=0, background="#f8f9fa", **kwargs)
        self.command = command
        self.bg_color = bg_color
        self.hover_color = hover_color
        self.corner_radius = corner_radius
        self.current_color = bg_color
        
        self.inner_pad = 3
      
        

        # Draw button
        self.draw_button(text, fg_color)
        
        # Bind events
        self.bind("<Enter>", self.on_enter)
        self.bind("<Leave>", self.on_leave)
        self.bind("<Button-1>", self.on_click)
        self.bind("<ButtonRelease-1>", self.on_release)
    
    def draw_button(self, text, fg_color):
        self.delete("all")
        pad = self.inner_pad
        width = self.winfo_reqwidth()
        height = self.winfo_reqheight()
        # Draw rounded rectangle
        self.create_rounded_rect(pad, pad, width - pad, height - pad, 
                                 self.corner_radius, fill=self.current_color)
        
        # Draw text
        self.create_text(width / 2, height / 2, 
                         text=text, fill=fg_color, font=("Montserrat", 10, "bold"))
    
    def create_rounded_rect(self, x1, y1, x2, y2, radius, **kwargs):
        # Create rounded rectangle
        points = [
            x1+radius, y1,
            x2-radius, y1,
            x2, y1,
            x2, y1+radius,
            x2, y2-radius,
            x2, y2,
            x2-radius, y2,
            x1+radius, y2,
            x1, y2,
            x1, y2-radius,
            x1, y1+radius,
            x1, y1
            
        ]
        return self.create_polygon(points, smooth=True, **kwargs)
    
    def on_enter(self, event):
        self.current_color = self.hover_color
        self.itemconfig(1, fill=self.current_color)
    
    def on_leave(self, event):
        self.current_color = self.bg_color
        self.itemconfig(1, fill=self.current_color)
    
    def on_click(self, event):
        self.itemconfig(1, fill=self.hover_color)
    
    def on_release(self, event):
        self.itemconfig(1, fill=self.hover_color)
        if self.command:
            self.command()

class GradientFrame(tk.Canvas):
    def __init__(self, parent, color1="#2c3e50", color2="#4a86e8", **kwargs):
        tk.Canvas.__init__(self, parent, **kwargs)
        self._color1 = color1
        self._color2 = color2
        self.bind("<Configure>", self._draw_gradient)

    def _draw_gradient(self, event=None):
        self.delete("gradient")
        width = self.winfo_width()
        height = self.winfo_height()
        
        # Create gradient
        limit = width
        (r1, g1, b1) = self.winfo_rgb(self._color1)
        (r2, g2, b2) = self.winfo_rgb(self._color2)
        r_ratio = float(r2-r1) / limit
        g_ratio = float(g2-g1) / limit
        b_ratio = float(b2-b1) / limit

        for i in range(limit):
            nr = int(r1 + (r_ratio * i))
            ng = int(g1 + (g_ratio * i))
            nb = int(b1 + (b_ratio * i))
            color = "#%4.4x%4.4x%4.4x" % (nr, ng, nb)
            self.create_line(i, 0, i, height, tags=("gradient",), fill=color)
        
        self.lower("gradient")

class TSPVisualizer:
    def __init__(self, root):
        self.root = root
        self.root.title("TSP Algorithm Visualizer")
        self.root.geometry("1280x800")
        self.root.configure(bg="#f8f9fa")
        
        # Set custom fonts
        self.title_font = font.Font(family="Montserrat", size=18, weight="bold")
        self.header_font = font.Font(family="Montserrat", size=12, weight="bold")
        self.text_font = font.Font(family="Montserrat", size=10)
        
        # Color scheme
        self.colors = {
            "bg": "#f8f9fa",
            "panel": "#ffffff",
            "accent": "#4a86e8",
            "accent_dark": "#3a76d8",
            "text": "#333333",
            "text_light": "#666666",
            "border": "#e1e4e8",
            "success": "#28a745",
            "warning": "#ffc107",
            "danger": "#dc3545",
            "brute_force": "#e74c3c",
            "nearest_neighbor": "#2ecc71",
            "genetic": "#9b59b6"
        }
        
        # Variables
        self.cities = []
        self.current_algorithm = tk.StringVar(value="Brute Force")
        self.num_cities = tk.IntVar(value=5)
        self.population_size = tk.IntVar(value=50)
        self.generations = tk.IntVar(value=100)
        self.mutation_rate = tk.DoubleVar(value=0.1)
        self.animation = None
        self.solving_time = 0
        
        # Configure ttk style
        self.configure_style()
        
        # Create main frames
        self.create_frames()
        self.create_header()
        self.create_controls()
        self.create_visualization()
        
        # Welcome message
        self.show_welcome_screen()
    
    def configure_style(self):
        self.style = ttk.Style()
        self.style.theme_use("clam")
        
        # Configure common elements
        self.style.configure("TFrame", background=self.colors["bg"])
        self.style.configure("Panel.TFrame", background=self.colors["panel"], 
                            relief="flat", borderwidth=0)
        
        self.style.configure("TLabel", background=self.colors["bg"], 
                            foreground=self.colors["text"], font=self.text_font)
        self.style.configure("Header.TLabel", font=self.header_font)
        self.style.configure("Title.TLabel", font=self.title_font)
        
        self.style.configure("TButton", background=self.colors["accent"], 
                            foreground="white", font=self.text_font)
        
        self.style.configure("TRadiobutton", background=self.colors["panel"], 
                            foreground=self.colors["text"], font=self.text_font)
        
        self.style.configure("TScale", background=self.colors["panel"])
        
        # Configure specific elements (without border for inner Labelframes)
        self.style.configure("Panel.TLabelframe", background=self.colors["panel"], 
                            relief="flat", borderwidth=0)
        self.style.configure("Panel.TLabelframe.Label", background=self.colors["panel"], 
                            foreground=self.colors["text"], font=self.header_font)
        
        # Configure scrollbar width
        self.style.configure("Vertical.TScrollbar", width=15)
    
    def create_frames(self):
        # Main container with padding
        self.main_frame = ttk.Frame(self.root, style="TFrame", padding=15)
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Header frame (no border) with more padding
        self.header_frame = ttk.Frame(self.main_frame, style="TFrame", padding=(0, 0, 0, 20))
        self.header_frame.pack(fill=tk.X, pady=(0, 15))
        
        # Content frame with control panel and visualization
        self.content_frame = ttk.Frame(self.main_frame, style="TFrame")
        self.content_frame.pack(fill=tk.BOTH, expand=True)
        
        # Control panel on the left with scrollbar and border
        self.control_container = tk.Frame(self.content_frame, 
                                         background=self.colors["bg"],
                                         highlightthickness=1, 
                                         highlightbackground="#d3d3d3", 
                                         width=230)
        self.control_container.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 15))
        self.control_container.pack_propagate(False)
        
        # Create canvas with a fixed height to ensure scrollbar appears
        self.control_canvas = tk.Canvas(self.control_container, highlightthickness=0, bg=self.colors["bg"], width=200, height=600)
        
        # Create scrollbar with a wrapper frame to control its height
        self.scrollbar_wrapper = ttk.Frame(self.control_container, width=20, height=200)
        self.scrollbar_wrapper.pack(side="right", fill="y", expand=False)
        self.scrollbar_wrapper.pack_propagate(False)
        
        # Center the scrollbar vertically within the wrapper frame
        self.scrollbar = ttk.Scrollbar(self.scrollbar_wrapper, orient="vertical", command=self.control_canvas.yview)
        self.scrollbar.pack(side="right", fill="y", pady=(0, 0))
        
        self.scrollable_frame = ttk.Frame(self.control_canvas, style="TFrame")
        
        # Configure the canvas
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.control_canvas.configure(
                scrollregion=self.control_canvas.bbox("all")
            )
        )
        
        self.control_canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.control_canvas.configure(yscrollcommand=self.scrollbar.set)
        
        # Pack the canvas with a small gap on the right to separate it from the scrollbar
        self.control_canvas.pack(side="left", fill="both", expand=False, padx=(0, 5))
        
        # Bind mousewheel for scrolling
        self.control_canvas.bind_all("<MouseWheel>", self._on_mousewheel)
        
        # Now create the control frame inside the scrollable frame
        self.control_frame = ttk.Frame(self.scrollable_frame, style="Panel.TFrame", padding=15)
        self.control_frame.pack(fill=tk.BOTH, expand=True)
        
        # Visualization panel on the right with border
        self.viz_container = tk.Frame(self.content_frame,
                                     background=self.colors["panel"],
                                     highlightthickness=1,  
                                     highlightbackground="#d3d3d3")  
        self.viz_container.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        self.viz_frame = ttk.Frame(self.viz_container, style="Panel.TFrame", padding=15)
        self.viz_frame.pack(fill=tk.BOTH, expand=True)
    
    def _on_mousewheel(self, event):
        # Reduce scroll distance by scaling the delta
        scroll_step = int(-1 * (event.delta / 120) * 0.5)  # Scale by 0.5 to make scrolling shorter
        if scroll_step != 0:  # Only scroll if the step is non-zero
            self.control_canvas.yview_scroll(scroll_step, "units")
    
    def create_header(self):
        # Logo/Title area with more padding
        logo_frame = ttk.Frame(self.header_frame, style="TFrame", padding=(0, 0, 20, 0))
        logo_frame.pack(side=tk.LEFT, fill=tk.Y)
        
        # Create a simple logo
        logo_canvas = tk.Canvas(logo_frame, width=50, height=50, 
                               background=self.colors["bg"], highlightthickness=0)
        logo_canvas.pack(side=tk.LEFT, padx=(0, 15))
        
        # Draw a simple TSP-like path as logo
        points = [(10, 25), (20, 10), (30, 40), (40, 20), (10, 25)]
        logo_canvas.create_polygon(points, outline=self.colors["accent"], 
                                  width=2, fill="", smooth=True)
        for x, y in points[:-1]:
            logo_canvas.create_oval(x-4, y-4, x+4, y+4, 
                                   fill=self.colors["accent"], outline="")
        
        # Title 
        title_label = ttk.Label(logo_frame, text="TSP Algorithm Visualizer", 
                               style="Title.TLabel", font=("Montserrat", 16, "bold"))
        title_label.pack(side=tk.LEFT)
        
        # Status area on the right
        status_frame = ttk.Frame(self.header_frame, style="TFrame")
        status_frame.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Status indicators
        self.status_var = tk.StringVar(value="Ready")
        self.time_var = tk.StringVar(value="Time: 0.00s")
        self.distance_var = tk.StringVar(value="Distance: 0.00")
        
        # Font for status indicators
        status_font = font.Font(family="Montserrat", size=12, weight="bold")
        
        # Status label with larger font
        status_label = ttk.Label(status_frame, textvariable=self.status_var, 
                                foreground=self.colors["success"],
                                font=status_font)
        status_label.pack(anchor=tk.E, pady=2)
        
        # Time label with larger font
        time_label = ttk.Label(status_frame, textvariable=self.time_var,
                              font=status_font)
        time_label.pack(anchor=tk.E, pady=2)
        
        # Distance label with larger font
        distance_label = ttk.Label(status_frame, textvariable=self.distance_var,
                                 font=status_font)
        distance_label.pack(anchor=tk.E, pady=2)
    
    def create_controls(self):
        # Remove placeholder content before adding actual controls
        for widget in self.control_frame.winfo_children():
            widget.destroy()
        
        # Algorithm selection panel
        algo_frame = ttk.LabelFrame(self.control_frame, text="Select Algorithm", 
                                   style="Panel.TLabelframe", padding=15)
        algo_frame.pack(fill=tk.X, pady=(0, 15))
        
        # Algorithm icons and descriptions
        algorithms = [
            ("Brute Force", self.colors["brute_force"]),
            ("Nearest Neighbor", self.colors["nearest_neighbor"]),
            ("Genetic Algorithm", self.colors["genetic"])
        ]
        
        for algo, color in algorithms:
            algo_container = ttk.Frame(algo_frame, style="Panel.TFrame")
            algo_container.pack(fill=tk.X, pady=5)
            
            # Color indicator
            color_indicator = tk.Canvas(algo_container, width=15, height=15, 
                                       background=self.colors["panel"], highlightthickness=0)
            color_indicator.pack(side=tk.LEFT, padx=(0, 10))
            color_indicator.create_oval(0, 0, 15, 15, fill=color, outline="")
            
            # Algorithm info
            algo_info = ttk.Frame(algo_container, style="Panel.TFrame")
            algo_info.pack(side=tk.LEFT, fill=tk.X, expand=True)
            
            rb = ttk.Radiobutton(algo_info, text=algo, value=algo, 
                                variable=self.current_algorithm)
            rb.pack(anchor=tk.W)
        
        # Parameters panel
        param_frame = ttk.LabelFrame(self.control_frame, text="Parameters", 
                                    style="Panel.TLabelframe", padding=15)
        param_frame.pack(fill=tk.X, pady=(0, 15))
        
        # Number of cities with fancy slider
        city_container = ttk.Frame(param_frame, style="Panel.TFrame")
        city_container.pack(fill=tk.X, pady=5)
        
        city_label = ttk.Label(city_container, text="Number of Cities:", 
                              style="Header.TLabel")
        city_label.pack(anchor=tk.W, pady=(0, 5))
        
        city_value_frame = ttk.Frame(city_container, style="Panel.TFrame")
        city_value_frame.pack(fill=tk.X)
        
        # Function to ensure int values for city count
        def on_city_scale_changed(event):
            # Get the current value and round it to an integer
            value = city_scale.get()
            int_value = round(value)
            # Set the value only if it's different to avoid infinite loop
            if value != int_value:
                city_scale.set(int_value)
                self.num_cities.set(int_value)
        
        city_scale = ttk.Scale(city_value_frame, from_=4, to=20, 
                              variable=self.num_cities, orient=tk.HORIZONTAL)
        city_scale.bind("<ButtonRelease-1>", on_city_scale_changed)
        city_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))
        
        city_value = ttk.Label(city_value_frame, textvariable=self.num_cities, 
                              width=2, anchor=tk.CENTER)
        city_value.pack(side=tk.RIGHT)
        
        # GA parameters (only shown for Genetic Algorithm)
        self.ga_frame = ttk.Frame(param_frame, style="Panel.TFrame")
        
        # Population size
        pop_container = ttk.Frame(self.ga_frame, style="Panel.TFrame")
        pop_container.pack(fill=tk.X, pady=5)
        
        pop_label = ttk.Label(pop_container, text="Population Size:", 
                             style="Header.TLabel")
        pop_label.pack(anchor=tk.W, pady=(0, 5))
        
        pop_value_frame = ttk.Frame(pop_container, style="Panel.TFrame")
        pop_value_frame.pack(fill=tk.X)
        
        pop_scale = ttk.Scale(pop_value_frame, from_=10, to=200, 
                             variable=self.population_size, orient=tk.HORIZONTAL)
        pop_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))
        
        pop_value = ttk.Label(pop_value_frame, textvariable=self.population_size, 
                             width=3, anchor=tk.CENTER)
        pop_value.pack(side=tk.RIGHT)
        
        # Generations
        gen_container = ttk.Frame(self.ga_frame, style="Panel.TFrame")
        gen_container.pack(fill=tk.X, pady=5)
        
        gen_label = ttk.Label(gen_container, text="Generations:", 
                             style="Header.TLabel")
        gen_label.pack(anchor=tk.W, pady=(0, 5))
        
        gen_value_frame = ttk.Frame(gen_container, style="Panel.TFrame")
        gen_value_frame.pack(fill=tk.X)
        
        gen_scale = ttk.Scale(gen_value_frame, from_=10, to=500, 
                             variable=self.generations, orient=tk.HORIZONTAL)
        gen_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))
        
        gen_value = ttk.Label(gen_value_frame, textvariable=self.generations, 
                             width=3, anchor=tk.CENTER)
        gen_value.pack(side=tk.RIGHT)
        
        # Mutation rate
        mut_container = ttk.Frame(self.ga_frame, style="Panel.TFrame")
        mut_container.pack(fill=tk.X, pady=5)
        
        mut_label = ttk.Label(mut_container, text="Mutation Rate:", 
                             style="Header.TLabel")
        mut_label.pack(anchor=tk.W, pady=(0, 5))
        
        mut_value_frame = ttk.Frame(mut_container, style="Panel.TFrame")
        mut_value_frame.pack(fill=tk.X)
        
        mut_scale = ttk.Scale(mut_value_frame, from_=0.01, to=0.5, 
                             variable=self.mutation_rate, orient=tk.HORIZONTAL)
        mut_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))
        
        mut_value = ttk.Label(mut_value_frame, textvariable=self.mutation_rate, 
                             width=4, anchor=tk.CENTER)
        mut_value.pack(side=tk.RIGHT)
        
        # Action buttons with fancy styling
        btn_frame = ttk.Frame(self.control_frame, style="Panel.TFrame")
        btn_frame.pack(fill=tk.X, pady=(0, 15))
        
        # Generate Cities button
        generate_btn = ModernButton(btn_frame, text="Generate Cities", 
                                   command=self.generate_random_cities, 
                                   width=180, height=40, 
                                   bg_color=self.colors["accent"], 
                                   hover_color=self.colors["accent_dark"])
        generate_btn.pack(fill=tk.X, pady=5)
        
        # Run Algorithm button
        run_btn = ModernButton(btn_frame, text="Run Algorithm", 
                              command=self.run_algorithm, 
                              width=180, height=40, 
                              bg_color=self.colors["success"], 
                              hover_color="#218838")
        run_btn.pack(fill=tk.X, pady=5)
        
        # Stop button
        stop_btn = ModernButton(btn_frame, text="Stop", 
                               command=self.stop_animation, 
                               width=180, height=40, 
                               bg_color=self.colors["danger"], 
                               hover_color="#c82333")
        stop_btn.pack(fill=tk.X, pady=5)
        
        # Clear button
        clear_btn = ModernButton(btn_frame, text="Clear", 
                                command=self.clear, 
                                width=180, height=40, 
                                bg_color=self.colors["warning"], 
                                hover_color="#e0a800",
                                fg_color=self.colors["text"])
        clear_btn.pack(fill=tk.X, pady=5)
        
        # Update GA frame visibility based on algorithm selection
        self.current_algorithm.trace_add("write", self.update_ga_visibility)
    
    def update_ga_visibility(self, *args):
        if self.current_algorithm.get() == "Genetic Algorithm":
            self.ga_frame.pack(fill=tk.X, pady=10)
        else:
            self.ga_frame.pack_forget()
    
    def create_visualization(self):
        # Create matplotlib figure with custom styling
        plt.style.use('seaborn-v0_8-whitegrid')
        self.fig, self.ax = plt.subplots(figsize=(8, 6), dpi=100)
        self.fig.patch.set_facecolor(self.colors["panel"])
        
        # Embed in tkinter
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.viz_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Configure plot
        self.ax.set_title("TSP Visualization", fontsize=16, fontweight='bold')
        self.ax.set_xlabel("X Coordinate", fontsize=12)
        self.ax.set_ylabel("Y Coordinate", fontsize=12)
        self.ax.grid(True, linestyle='--', alpha=0.7)
        self.ax.spines['top'].set_visible(False)
        self.ax.spines['right'].set_visible(False)
    
    def show_welcome_screen(self):
        self.ax.clear()
        
        # Scale sizes based on figure dimensions
        fig_width, fig_height = self.fig.get_size_inches()
        scale = min(fig_width/8, fig_height/6)
        font_size = 14 * scale
        city_size = 10 * scale
        title_size = 18 * scale
        
        # Welcome text
        welcome_text = (
            "Welcome to TSP Algorithm Visualizer\n\n"
            "Explore different algorithms for solving the\n"
            "Traveling Salesman Problem\n\n"
            "1. Select an algorithm from the left panel\n"
            "2. Set the number of cities and parameters\n"
            "3. Generate random cities\n"
            "4. Run the algorithm to visualize the solution"
        )
        
        # Add a background pattern
        x = np.linspace(0, 10, 1000)
        y = np.linspace(0, 10, 1000)
        X, Y = np.meshgrid(x, y)
        Z = np.sin(X) * np.cos(Y)
        
        # Plot with a light colormap
        c = self.ax.pcolormesh(X, Y, Z, cmap='Blues', alpha=0.1, shading='auto')
        
        # Check if Montserrat is available, else use Arial
        import matplotlib.font_manager
        available_fonts = [f.name for f in matplotlib.font_manager.fontManager.ttflist]
        font_family = 'Montserrat' if 'Montserrat' in available_fonts else 'Arial'
        
        # Add welcome text with custom font and alignment
        self.ax.text(0.5, 0.52, welcome_text, 
                    horizontalalignment='center',
                    verticalalignment='center',
                    fontsize=font_size,
                    fontfamily=font_family,
                    fontweight='bold',
                    color='#000000',            
                    transform=self.ax.transAxes,
                    bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=1', 
                            edgecolor='#e1e4e8'))
        
        # Add a simple TSP example in the background
        cities = [(2, 2), (8, 2), (8, 8), (2, 8)]
        x = [city[0] for city in cities]
        y = [city[1] for city in cities]
        x.append(x[0])  # Close the loop
        y.append(y[0])
        
        self.ax.plot(x, y, 'o-', color=self.colors["accent"], alpha=0.5, 
                    markersize=city_size, linewidth=2)
        
        self.ax.set_xlim(0, 10)
        self.ax.set_ylim(0, 10)
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.ax.set_title("TSP Algorithm Visualizer", fontsize=title_size, fontweight='bold')
        
        self.canvas.draw()
    
    def generate_random_cities(self):
        n = self.num_cities.get()
        self.cities = [(random.uniform(0, 100), random.uniform(0, 100)) for _ in range(n)]
        
        # Clear any existing animation
        self.stop_animation()
        
        # Clear the existing figure completely
        self.fig.clf()
        self.ax = self.fig.add_subplot(111)
        
        # Plot cities
        self.plot_cities()
        self.status_var.set(f"Generated {n} random cities")
    
    def plot_cities(self):
        self.ax.clear()
        x = [city[0] for city in self.cities]
        y = [city[1] for city in self.cities]
        
        # Create a custom colormap for cities
        colors = [self.colors["accent"] for _ in range(len(self.cities))]
        
        # Plot cities with a more appealing style
        self.ax.scatter(x, y, s=120, c=colors, edgecolor='white', 
                       linewidth=2, zorder=10, alpha=0.8)
        
        # Add a subtle background grid with the same style as used in algorithms
        self.ax.grid(True, linestyle='--', alpha=0.3, color='gray')
        
        # Label cities with a nicer font
        for i, (x, y) in enumerate(self.cities):
            self.ax.annotate(str(i), (x, y), xytext=(0, 0), 
                            textcoords='offset points',
                            ha='center', va='center',
                            color='white', fontweight='bold',
                            fontsize=9)
        
        # Use the same axis limits as the algorithms will use
        self.ax.set_xlim(-5, 105)
        self.ax.set_ylim(-5, 105)
        self.ax.set_title("City Locations", fontsize=18, fontweight='bold')
        
        # Remove spines for a cleaner look
        self.ax.spines['top'].set_visible(False)
        self.ax.spines['right'].set_visible(False)
        
        self.canvas.draw()
    
    def run_algorithm(self):
        if not self.cities:
            messagebox.showwarning("Warning", "Please generate cities first!")
            return
        
        self.stop_animation()  # Stop any existing animation
        
        # Clear the existing figure completely
        self.fig.clf()
        
        algorithm = self.current_algorithm.get()
        self.status_var.set(f"Running {algorithm}...")
        
        # Reset any stored axes from genetic algorithm to avoid layout issues
        if hasattr(self, 'ax_tour'):
            delattr(self, 'ax_tour')
        if hasattr(self, 'ax_fitness'):
            delattr(self, 'ax_fitness')
        
        # For genetic algorithm, we don't create the axes here since it needs a special layout
        # with two subplots. For other algorithms, we create a single subplot.
        if algorithm != "Genetic Algorithm":
            self.ax = self.fig.add_subplot(111)
            # Reset figure layout to default (remove any previous adjustments)
            self.fig.subplots_adjust(left=0.125, right=0.9, bottom=0.1, top=0.9, wspace=0.2, hspace=0.2)
        
        if algorithm == "Brute Force":
            self.run_brute_force()
        elif algorithm == "Nearest Neighbor":
            self.run_nearest_neighbor()
        elif algorithm == "Genetic Algorithm":
            self.run_genetic_algorithm()
        else:
            messagebox.showinfo("Info", f"The {algorithm} algorithm is not implemented yet.")
    
    def run_brute_force(self):
        """Use the brute force visualization from the animation module"""
        # Stop any existing animation
        self.stop_animation()
        
        # Make sure we have cities to work with
        if not self.cities:
            messagebox.showwarning("Warning", "Please generate cities first!")
            return
        
        # Define custom styling that matches our application's style
        custom_style = {
            'fig_size': (8, 6),
            'background_color': self.colors["panel"],
            'plot_bg_color': '#f8f8f8',
            'city_color': self.colors["accent"],
            'path_color': self.colors["brute_force"],
            'best_path_color': self.colors["success"],
            'new_best_color': self.colors["success"],
            'city_size': 120,  # Exactly match the size used in plot_cities
            'city_edge_color': 'white',
            'city_edge_width': 2,
            'city_alpha': 0.8,
            'font_size': 9,
            'animation_interval': 1000,
            'repeat': False,
            'grid_alpha': 0.3,
            'grid_color': 'gray',
            'grid_linestyle': '--',
            'x_min': -5,
            'x_max': 105,
            'y_min': -5,
            'y_max': 105,
            'title_size': 18,
            'title': "TSP Brute Force Solution",
            'label_size': 14,
            'legend_size': 8,
            'legend_loc': 'lower right'
        }
        
        # Update status for initial state
        self.status_var.set("Running Brute Force algorithm...")
        
        # Function to update status in the UI
        def update_status(title, distance, _):
            self.status_var.set(title)
            self.distance_var.set(f"Distance: {distance:.2f}")
            self.time_var.set(f"Time: {self.solving_time:.2f}s")
        
        # Import the visualization function from the animation module
        from animation.BruteForce import visualize_brute_force
        
        try:
            # Create a TSPSolver instance with the current cities
            tsp_solver = TSPSolver(coordinates=self.cities)
            
            # Run the brute force algorithm and measure the solving time
            start_time = time.time()
            path, distance = tsp_solver.solve_brute_force(visualize_steps=True)
            self.solving_time = time.time() - start_time  # This is the actual algorithm runtime
            
            # Use the external visualization function, but with our figure/axes
            ani = visualize_brute_force(
                coordinates=self.cities,
                custom_style=custom_style,
                show_plot=False,
                fig=self.fig,
                ax=self.ax,
                update_status_callback=update_status
            )
            
            # Store the animation object so we can stop it later
            self.animation = ani
            
            # Update the matplotlib figure in our canvas
            self.canvas.draw()
            
            # Enable resizing support - redraw canvas if window is resized
            def on_resize(event):
                self.canvas.draw_idle()
            
            self.fig.canvas.mpl_connect('resize_event', on_resize)
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to run Brute Force: {str(e)}")
            self.status_var.set("Ready")
    
    def run_nearest_neighbor(self):
        """Use the nearest neighbor visualization from the animation module"""
        # Stop any existing animation
        self.stop_animation()
        
        # Make sure we have cities to work with
        if not self.cities:
            messagebox.showwarning("Warning", "Please generate cities first!")
            return
        
        # Start with default style and update with application colors
        custom_style = DEFAULT_STYLE.copy()
        custom_style.update({
            'background_color': self.colors["panel"],
            'city_color': self.colors["accent"],
            'path_color': self.colors["nearest_neighbor"],
            'next_city_color': self.colors["success"]
        })
        
        # Update status for initial state
        self.status_var.set("Running Nearest Neighbor algorithm...")
        
        # Function to update status in the UI
        def update_status(title, distance, time):
            self.status_var.set(title)
            self.distance_var.set(f"Distance: {distance:.2f}")
            self.time_var.set(f"Time: {time:.2f}s")
            # Store the solving time
            self.solving_time = time
        
        # Import the visualization function from the animation module
        from animation.NearestNeighbor import visualize_nearest_neighbor
        
        try:
            # Create a TSPSolver instance with the current cities
            tsp_solver = TSPSolver(coordinates=self.cities)
            
            # Run the nearest neighbor algorithm and measure the solving time
            start_time = time.time()
            path, distance = tsp_solver.solve_nearest_neighbor(visualize_steps=True)
            self.solving_time = time.time() - start_time  # This is the actual algorithm runtime
            
            # Use the external visualization function, but with our figure/axes
            ani = visualize_nearest_neighbor(
                coordinates=self.cities,
                custom_style=custom_style,
                show_plot=False,
                fig=self.fig,
                ax=self.ax,
                update_status_callback=update_status
            )
            
            # Store the animation object so we can stop it later
            self.animation = ani
            
            # Update the matplotlib figure in our canvas
            self.canvas.draw()
            
            # Enable resizing support - redraw canvas if window is resized
            def on_resize(event):
                self.canvas.draw_idle()
            
            self.fig.canvas.mpl_connect('resize_event', on_resize)
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to run Nearest Neighbor: {str(e)}")
            self.status_var.set("Ready")
    
    def run_genetic_algorithm(self):
        """Run and visualize the genetic algorithm for TSP with animation matching the provided animate_tsp style"""
        # Stop any existing animation
        self.stop_animation()
        
        # Make sure we have cities to work with
        if not self.cities:
            messagebox.showwarning("Warning", "Please generate cities first!")
            return
        
        # Define custom styling that matches our application's style
        custom_style = {
            'fig_size': (10, 6), 
            'background_color': self.colors["panel"],
            'plot_bg_color': '#f8f8f8',
            'city_color': self.colors["accent"],
            'path_color': 'red',  
            'current_city_color': 'green',
            'city_size': 80,  
            'city_edge_color': 'white',
            'city_edge_width': 2,
            'city_alpha': 0.8,
            'font_size': 10,
            'animation_interval': 200,
            'repeat': False,
            'grid_alpha': 0.3,
            'grid_color': 'gray',
            'grid_linestyle': '--',
            'x_min': -5,
            'x_max': 105,
            'y_min': -5,
            'y_max': 105,
            'title_size': 16,
            'title': "TSP Genetic Algorithm Solution",
            'label_size': 12,
            'legend_size': 9,
            'legend_loc': 'lower right'
        }

        # Update status for initial state
        self.status_var.set("Running Genetic Algorithm...")
        
        # Function to update status in the UI
        def update_status(title, distance, elapsed_time):
            self.status_var.set(title)
            self.distance_var.set(f"Distance: {distance:.2f}")
            self.time_var.set(f"Time: {elapsed_time:.2f}s")

        try:
            # Create a TSPSolver instance with the current cities
            tsp_solver = TSPSolver(coordinates=self.cities)
            
            # Run the genetic algorithm and measure only the solving time
            start_time = time.time()
            tsp_solver.solve_genetic(
                pop_size=self.population_size.get(),
                generations=self.generations.get(),
                mutation_rate=self.mutation_rate.get(),
                visualize_steps=True
            )
            self.solving_time = time.time() - start_time  # This is the actual algorithm runtime
            
            # Store the best distance
            self.best_distance = min(tsp_solver.path_distances)

            # Check if paths are available for animation
            if not tsp_solver.all_paths:
                raise ValueError("No paths available for animation. Ensure solve_genetic was run with visualize_steps=True.")
            
            # Create the subplots for genetic algorithm
            # Note: The figure was already cleared in run_algorithm
            gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1])
            self.ax_tour = self.fig.add_subplot(gs[0])
            self.ax_fitness = self.fig.add_subplot(gs[1])
            
            # Adjust subplot spacing to make more room for labels and improve layout
            self.fig.subplots_adjust(left=0.1, right=0.9, bottom=0.15, top=0.9, wspace=0.3)
            
            # Calculate frames: one per discovered path
            total_frames = len(tsp_solver.all_paths)
            
            # Animation update function
            def update(frame):
                # Clear all plots for the new frame
                self.ax_tour.clear()
                self.ax_fitness.clear()
                
                # Get the current path and its generation
                path_idx = frame
                path = tsp_solver.all_paths[path_idx]
                path_coords = tsp_solver.coordinates[list(path)]
                gen_idx = tsp_solver.path_generations[path_idx]
                # Use a consistent color for the path instead of the gradient
                color = custom_style['path_color']

                # Plot all cities
                self.ax_tour.scatter(tsp_solver.coordinates[:, 0], tsp_solver.coordinates[:, 1], 
                                     c=custom_style['city_color'], s=custom_style['city_size'], 
                                     edgecolors=custom_style['city_edge_color'], 
                                     linewidths=custom_style['city_edge_width'], 
                                     alpha=custom_style['city_alpha'], label='Cities')
                
                # Draw the current path
                closed_path_coords = np.vstack([path_coords, path_coords[0]])
                self.ax_tour.plot(closed_path_coords[:, 0], closed_path_coords[:, 1], 
                              color=color, linestyle='-', linewidth=2, label='Current Path',
                              zorder=2)

                # Only show the start city (green star) if this is not the final frame
                if frame != total_frames - 1:
                    current_city = path_coords[0]
                    self.ax_tour.scatter(current_city[0], current_city[1], c=custom_style['current_city_color'], 
                                         s=custom_style['city_size'] * 1.5, marker='*', label='Start City',
                                         zorder=3)

                # Update title with generation and distance
                self.ax_tour.set_xlabel('X Coordinate', fontsize=custom_style['label_size'])
                self.ax_tour.set_ylabel('Y Coordinate', fontsize=custom_style['label_size'])
                self.ax_tour.grid(True, linestyle=custom_style['grid_linestyle'], 
                                  alpha=custom_style['grid_alpha'], color=custom_style['grid_color'])
                self.ax_tour.legend(fontsize=custom_style['legend_size'], loc=custom_style['legend_loc'])

                # Update fitness plot (convergence)
                generations = list(range(len(tsp_solver.generation_best_distances)))
                distances = tsp_solver.generation_best_distances
                self.ax_fitness.plot(generations, distances, 'b-', label='Best Distance')
                # Always show the red dot for the current generation
                self.ax_fitness.scatter([gen_idx], [distances[gen_idx]], c='red', s=custom_style['city_size'], 
                                       marker='o', label='Current Generation')
                self.ax_fitness.set_xlabel('Generation', fontsize=custom_style['label_size'])
                self.ax_fitness.set_ylabel('Distance', fontsize=custom_style['label_size'])
                self.ax_fitness.grid(True, linestyle=custom_style['grid_linestyle'], 
                                     alpha=custom_style['grid_alpha'], color=custom_style['grid_color'])
                self.ax_fitness.legend(fontsize=custom_style['legend_size'], loc=custom_style['legend_loc'])
                
                # Update dynamic status
                distance = tsp_solver.path_distances[path_idx]
                if frame == total_frames - 1:
                    update_status("Best Solution Found", self.best_distance, self.solving_time)
                else:
                    update_status(f"Generation {gen_idx+1}", distance, self.solving_time)

                # Set consistent axis limits
                self.ax_tour.set_xlim(custom_style['x_min'], custom_style['x_max'])
                self.ax_tour.set_ylim(custom_style['y_min'], custom_style['y_max'])

            # Create the animation
            ani = FuncAnimation(self.fig, update, frames=total_frames, 
                               interval=custom_style['animation_interval'], repeat=custom_style['repeat'])
            
            # Store the animation object so we can stop it later
            self.animation = ani
            
            # Update the matplotlib figure in our canvas
            self.canvas.draw()
            
            # Enable resizing support - redraw canvas if window is resized
            def on_resize(event):
                self.canvas.draw_idle()
            
            self.fig.canvas.mpl_connect('resize_event', on_resize)
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to run Genetic Algorithm: {str(e)}")
            self.status_var.set("Ready")
    
    def stop_animation(self):
        if self.animation:
            self.animation.event_source.stop()
            self.animation = None
    
    def clear(self):
        self.stop_animation()
        self.cities = []
        self.status_var.set("Ready")
        self.time_var.set("Time: 0.00s")
        self.distance_var.set("Distance: 0.00")
        
        # Clear the figure and create a new single subplot
        self.fig.clf()
        self.ax = self.fig.add_subplot(111)
        
        # Reset any stored axes from genetic algorithm
        if hasattr(self, 'ax_tour'):
            delattr(self, 'ax_tour')
        if hasattr(self, 'ax_fitness'):
            delattr(self, 'ax_fitness')
        
        # Reset figure layout to default
        self.fig.subplots_adjust(left=0.125, right=0.9, bottom=0.1, top=0.9, wspace=0.2, hspace=0.2)
            
        self.show_welcome_screen()

# Run the application
if __name__ == "__main__":
    root = tk.Tk()
    app = TSPVisualizer(root)
    root.mainloop()