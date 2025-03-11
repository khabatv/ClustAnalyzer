import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler, QuantileTransformer, PowerTransformer, Normalizer
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist, squareform
import numpy as np
import os
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog, ttk, colorchooser
import json

def correlation_distance(X, metric='pearson'):
    if metric == 'pearson':
        corr = np.corrcoef(X)
    elif metric == 'spearman':
        corr = pd.DataFrame(X).T.corr(method='spearman').values
    else:
        raise ValueError("Unknown metric. Use 'pearson' or 'spearman'.")
    dist = 1 - corr
    return squareform(dist, checks=False)

def z_score_normalization(data):
    return (data - data.mean()) / data.std()

def pareto_scaling(data):
    return (data - data.mean()) / np.sqrt(data.std())

def mean_centering(data):
    return data - data.mean()

def log_transformation(data):
    return np.log1p(data)

def l1_normalization(data):
    return data / np.abs(data).sum(axis=0)

def unit_vector_normalization(data):
    return Normalizer().fit_transform(data)

class ClusteringGUI:
    def __init__(self, master):
        self.master = master
        self.master.title("Clustering Analysis")
        self.master.geometry("400x600")

        self.data_file = ""
        self.design_file = ""
        self.output_dir = ""
        self.condition_colors = {}
        self.default_colors = [
            "#FF0000", "#00FF00", "#0000FF", "#FFFF00", "#FF00FF", "#00FFFF",
            "#800000", "#008000", "#000080", "#808000", "#800080", "#008080",
            "#FFA500", "#A52A2A", "#FFC0CB", "#90EE90", "#ADD8E6", "#E6E6FA",
            "#8B4513", "#4B0082", "#7FFFD4", "#D2691E", "#DC143C", "#556B2F",
            "#B8860B", "#32CD32", "#9932CC", "#F0E68C", "#00CED1", "#FFFFFF"
        ]
        self.custom_heatmap_colors = []

        self.create_widgets()

    def create_widgets(self):
        # File selection buttons
        tk.Button(self.master, text="Select Data File", command=self.select_data_file).pack(pady=5)
        tk.Button(self.master, text="Select Design File", command=self.select_design_file).pack(pady=5)
        tk.Button(self.master, text="Select Output Directory", command=self.select_output_dir).pack(pady=5)

        # KMeans clusters input
        tk.Label(self.master, text="Number of KMeans clusters:").pack(pady=5)
        self.num_clusters_var = tk.StringVar(value="5")
        tk.Entry(self.master, textvariable=self.num_clusters_var).pack()

        # Distance metric options
        tk.Label(self.master, text="Select Distance Metric:").pack(pady=5)
        self.distance_metric = tk.StringVar(value="euclidean")
        distance_options = ["euclidean", "pearson", "spearman"]
        tk.OptionMenu(self.master, self.distance_metric, *distance_options).pack()

        # Linkage method options
        tk.Label(self.master, text="Select Linkage Method:").pack(pady=5)
        self.linkage_method = tk.StringVar(value="average")
        linkage_options = ["average", "complete", "single"]
        tk.OptionMenu(self.master, self.linkage_method, *linkage_options).pack()

        # Heatmap color style options
        tk.Label(self.master, text="Select Heatmap Color Style:").pack(pady=5)
        self.heatmap_color_style = tk.StringVar(value="viridis")
        color_options = ["viridis", "plasma", "inferno", "magma", "cividis", "coolwarm", "RdBu", "Spectral", "cool", "hot", "custom"]
        tk.OptionMenu(self.master, self.heatmap_color_style, *color_options).pack()

        # Custom heatmap color selection
        tk.Button(self.master, text="Select Custom Heatmap Colors", command=self.select_custom_heatmap_colors).pack(pady=5)

        # Data normalization options
        tk.Label(self.master, text="Select Data Normalization Methods:").pack(pady=5)
        self.normalization_methods = tk.Listbox(self.master, selectmode=tk.MULTIPLE, exportselection=False)
        normalization_options = [
            "StandardScaler", "MinMaxScaler", "MaxAbsScaler", "RobustScaler",
            "QuantileTransformer", "PowerTransformer", "Z-Score Normalization",
            "Pareto Scaling", "Mean Centering", "Log Transformation",
            "L1 Normalization", "Unit Vector Normalization"
        ]
        for option in normalization_options:
            self.normalization_methods.insert(tk.END, option)
        self.normalization_methods.pack()

        # Button to set condition colors
        tk.Button(self.master, text="Set Condition Colors", command=self.set_condition_colors).pack(pady=10)

        # Buttons to save and load settings
        tk.Button(self.master, text="Save Settings", command=self.save_settings).pack(pady=5)
        tk.Button(self.master, text="Load Settings", command=self.load_settings).pack(pady=5)

        # Run button
        tk.Button(self.master, text="Run Analysis", command=self.run_analysis).pack(pady=10)

    def select_data_file(self):
        self.data_file = filedialog.askopenfilename(title="Select Data File", filetypes=[("Text files", "*.txt")])

    def select_design_file(self):
        self.design_file = filedialog.askopenfilename(title="Select Design File", filetypes=[("Text files", "*.txt")])

    def select_output_dir(self):
        self.output_dir = filedialog.askdirectory(title="Select Output Directory")

    def set_condition_colors(self):
        if not self.design_file:
            messagebox.showerror("Error", "Please select a design file first.")
            return

        design_raw = pd.read_csv(self.design_file, sep='\t', header=None)
        condition_names = design_raw.iloc[1:, 0].tolist()
        design_raw.columns = design_raw.iloc[0]
        design_data = design_raw.drop(0).drop(columns=[design_raw.columns[0]])
        design = design_data.T
        design.columns = condition_names

        for factor in design.columns:
            unique_vals = design[factor].dropna().unique()
            for val in unique_vals:
                if (factor, val) not in self.condition_colors:
                    color = colorchooser.askcolor(title=f"Choose color for {factor}: {val}")[1]
                    if color:
                        self.condition_colors[(factor, val)] = color

    def select_custom_heatmap_colors(self):
        self.custom_heatmap_colors = []
        num_colors = simpledialog.askinteger("Input", "How many colors do you want for the custom heatmap?", minvalue=2, maxvalue=10)
        if num_colors is not None:
            for _ in range(num_colors):
                color = colorchooser.askcolor(title="Choose a color for the heatmap")[1]
                if color:
                    self.custom_heatmap_colors.append(color)

    def run_analysis(self):
        if not self.data_file or not self.design_file or not self.output_dir:
            messagebox.showerror("Error", "Please select all required files and directory.")
            return

        try:
            num_clusters = int(self.num_clusters_var.get())
        except ValueError:
            messagebox.showerror("Error", "Invalid number of clusters.")
            return

        selected_normalizations = [self.normalization_methods.get(i) for i in self.normalization_methods.curselection()]
        if not selected_normalizations:
            messagebox.showerror("Error", "Please select at least one normalization method.")
            return

        self.perform_analysis(num_clusters, self.distance_metric.get(), self.linkage_method.get(), self.heatmap_color_style.get(), selected_normalizations)

    def perform_analysis(self, num_clusters, distance_metric, linkage_method, heatmap_color_style, normalization_methods):
        # Load expression data
        data = pd.read_csv(self.data_file, sep='\t', index_col=0)
        data = data.fillna(data.mean())

        # Load and process experimental design
        design_raw = pd.read_csv(self.design_file, sep='\t', header=None)
        condition_names = design_raw.iloc[1:, 0].tolist()
        design_raw.columns = design_raw.iloc[0]
        design_data = design_raw.drop(0).drop(columns=[design_raw.columns[0]])
        design = design_data.T
        design.columns = condition_names

        # Normalize the data
        data_scaled = data.copy()
        for method in normalization_methods:
            if method == "StandardScaler":
                scaler = StandardScaler()
                data_scaled = pd.DataFrame(scaler.fit_transform(data_scaled.T).T, index=data_scaled.index, columns=data_scaled.columns)
            elif method == "MinMaxScaler":
                scaler = MinMaxScaler()
                data_scaled = pd.DataFrame(scaler.fit_transform(data_scaled.T).T, index=data_scaled.index, columns=data_scaled.columns)
            elif method == "MaxAbsScaler":
                scaler = MaxAbsScaler()
                data_scaled = pd.DataFrame(scaler.fit_transform(data_scaled.T).T, index=data_scaled.index, columns=data_scaled.columns)
            elif method == "RobustScaler":
                scaler = RobustScaler()
                data_scaled = pd.DataFrame(scaler.fit_transform(data_scaled.T).T, index=data_scaled.index, columns=data_scaled.columns)
            elif method == "QuantileTransformer":
                scaler = QuantileTransformer()
                data_scaled = pd.DataFrame(scaler.fit_transform(data_scaled.T).T, index=data_scaled.index, columns=data_scaled.columns)
            elif method == "PowerTransformer":
                scaler = PowerTransformer()
                data_scaled = pd.DataFrame(scaler.fit_transform(data_scaled.T).T, index=data_scaled.index, columns=data_scaled.columns)
            elif method == "Z-Score Normalization":
                data_scaled = z_score_normalization(data_scaled)
            elif method == "Pareto Scaling":
                data_scaled = pareto_scaling(data_scaled)
            elif method == "Mean Centering":
                data_scaled = mean_centering(data_scaled)
            elif method == "Log Transformation":
                data_scaled = log_transformation(data_scaled)
            elif method == "L1 Normalization":
                data_scaled = l1_normalization(data_scaled)
            elif method == "Unit Vector Normalization":
                data_scaled = unit_vector_normalization(data_scaled)
            else:
                raise ValueError("Unknown normalization method.")

        # PCA for dimensionality reduction
        pca = PCA(n_components=min(50, data_scaled.shape[1]))
        data_pca = pca.fit_transform(data_scaled)

        # KMeans clustering
        kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
        kmeans_labels = kmeans.fit_predict(data_pca)
        data_scaled['KMeans_Cluster'] = kmeans_labels
        data_scaled = data_scaled.sort_values('KMeans_Cluster')
        sorted_labels = data_scaled['KMeans_Cluster'].values
        data_scaled = data_scaled.drop('KMeans_Cluster', axis=1)

        # Generate a filename based on input file and parameters
        base_filename = os.path.basename(self.data_file)[:10]
        norm_abbr = ''.join([method[:3].upper() for method in normalization_methods])
        dist_abbr = distance_metric[:3].upper()
        linkage_abbr = linkage_method[:3].upper()
        filename = f"{base_filename}_{norm_abbr}_{dist_abbr}_{linkage_abbr}"

        # Save KMeans cluster labels to a file
        kmeans_labels_df = pd.DataFrame(sorted_labels, index=data_scaled.index, columns=["KMeans_Cluster"])
        kmeans_labels_df.to_csv(os.path.join(self.output_dir, f'{filename}_KMeans_Clusters.txt'), sep='\t')

        # Limit rows to 1000 per cluster
        max_rows_per_cluster = 1000
        clustered_data = []
        for cluster in range(num_clusters):
            cluster_data = data_scaled[sorted_labels == cluster].iloc[:max_rows_per_cluster]
            clustered_data.append(cluster_data)
            if cluster < num_clusters - 1:
                blank_rows = pd.DataFrame(np.nan,
                                          index=[f'Blank_{cluster+1}_{i}' for i in range(30)],
                                          columns=data_scaled.columns)
                clustered_data.append(blank_rows)
        data_with_blanks = pd.concat(clustered_data)

        # Hierarchical clustering on samples
        if distance_metric in ['pearson', 'spearman']:
            dist_matrix = correlation_distance(data_scaled.T, metric=distance_metric)
            col_linkage = linkage(dist_matrix, method=linkage_method)
        else:
            col_linkage = linkage(data_scaled.T, method=linkage_method, metric=distance_metric)

        # Create color-coded column annotations
        col_colors = pd.DataFrame(index=data.columns, columns=design.columns)
        for factor in design.columns:
            col_colors[factor] = design[factor].reindex(data.columns).map(lambda x: self.condition_colors.get((factor, x), self.default_colors[hash(x) % len(self.default_colors)]))

        # Plot heatmap
        sns.set_context('notebook')
        if heatmap_color_style == "custom" and self.custom_heatmap_colors:
            cmap = sns.blend_palette(self.custom_heatmap_colors, as_cmap=True)
        else:
            cmap = heatmap_color_style

        g = sns.clustermap(data_with_blanks, row_cluster=False, col_linkage=col_linkage,
                           cmap=cmap, figsize=(12, 10), yticklabels=False,
                           col_colors=col_colors, cbar_kws={'label': 'Standardized Expression Level'})

        # Move title to the top of the plot
        g.fig.suptitle('KMeans Clustering of Features and Hierarchical Clustering of Samples', fontsize=16, y=1.05)

        # Adjust colorbar to fit within the plot and avoid overlap with labels
        g.cax.set_position([0.95, 0.2, 0.02, 0.5])

        # Correct vertical positioning of KMeans cluster labels
        cluster_sizes = [len(cluster) for cluster in clustered_data if not cluster.empty]
        cumulative_sizes = np.cumsum([0] + cluster_sizes)

        # Position cluster labels vertically so they align with their respective heatmap sections
        label_spacing = 0.05
        label_shift = 0.1

        for i in range(num_clusters):
            cluster_midpoint = (cumulative_sizes[i] + cumulative_sizes[i + 1]) / 2
            label_position = (cluster_midpoint / len(data_with_blanks)) + label_shift + (i * label_spacing)
            g.ax_heatmap.text(-0.05, label_position,
                              f'Cluster {i}', va='center', ha='right',
                              fontweight='bold', fontsize=12, transform=g.ax_heatmap.transAxes)

        # Create separate legends for each condition factor
        legend_vertical_spacing = 0.25
        for i, factor in enumerate(design.columns):
            unique_vals = list(design[factor].dropna().unique())
            handles = [mpatches.Patch(color=self.condition_colors.get((factor, val), self.default_colors[hash(val) % len(self.default_colors)]), label=str(val))
                       for val in unique_vals]
            leg = g.ax_heatmap.legend(handles=handles, title=factor, loc='upper left',
                                      bbox_to_anchor=(1.2, 1 - i * legend_vertical_spacing),
                                      borderaxespad=0.)
            g.fig.add_artist(leg)

        # Add parameter display (moved further to the right)
        param_text = f"KMeans clusters: {num_clusters}\nDistance metric: {distance_metric}\nLinkage method: {linkage_method}\nNormalization methods: {', '.join(normalization_methods)}"
        g.fig.text(1.2, 0.98, param_text, verticalalignment='top', horizontalalignment='right', fontsize=10, bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'))

        plt.tight_layout(rect=[0, 0, 0.85, 1])

        # Save the plot
        plt.savefig(os.path.join(self.output_dir, f'{filename}_Heatmap.png'), dpi=150, bbox_inches='tight')
        plt.show()  # Show the plot in Spyder
        plt.close()

        # Save settings
        settings = {
            "num_clusters": num_clusters,
            "distance_metric": distance_metric,
            "linkage_method": linkage_method,
            "heatmap_color_style": heatmap_color_style,
            "normalization_methods": normalization_methods,
            "condition_colors": self.condition_colors,
            "custom_heatmap_colors": self.custom_heatmap_colors
        }
        with open(os.path.join(self.output_dir, f'{filename}_settings.json'), 'w') as f:
            json.dump(settings, f)

        messagebox.showinfo("Success", f"Heatmap and settings saved in: {self.output_dir}")

    def save_settings(self):
        if not self.data_file or not self.output_dir:
            messagebox.showerror("Error", "Please select a data file and output directory first.")
            return

        # Generate a filename based on input file and parameters
        base_filename = os.path.basename(self.data_file)[:10]
        norm_abbr = ''.join([self.normalization_methods.get(i)[:3].upper() for i in self.normalization_methods.curselection()])
        dist_abbr = self.distance_metric.get()[:3].upper()
        linkage_abbr = self.linkage_method.get()[:3].upper()
        filename = f"{base_filename}_{norm_abbr}_{dist_abbr}_{linkage_abbr}"

        # Save settings
        settings = {
            "num_clusters": self.num_clusters_var.get(),
            "distance_metric": self.distance_metric.get(),
            "linkage_method": self.linkage_method.get(),
            "heatmap_color_style": self.heatmap_color_style.get(),
            "normalization_methods": [self.normalization_methods.get(i) for i in self.normalization_methods.curselection()],
            "condition_colors": self.condition_colors,
            "custom_heatmap_colors": self.custom_heatmap_colors
        }
        with open(os.path.join(self.output_dir, f'{filename}_settings.json'), 'w') as f:
            json.dump(settings, f)

        messagebox.showinfo("Success", f"Settings saved in: {self.output_dir}")

    def load_settings(self):
        settings_file = filedialog.askopenfilename(title="Select Settings File", filetypes=[("JSON files", "*.json")])
        if settings_file:
            with open(settings_file, 'r') as f:
                settings = json.load(f)

            self.num_clusters_var.set(settings["num_clusters"])
            self.distance_metric.set(settings["distance_metric"])
            self.linkage_method.set(settings["linkage_method"])
            self.heatmap_color_style.set(settings["heatmap_color_style"])
            self.normalization_methods.selection_clear(0, tk.END)
            for method in settings["normalization_methods"]:
                index = self.normalization_methods.get(0, tk.END).index(method)
                self.normalization_methods.selection_set(index)
            self.condition_colors = settings["condition_colors"]
            self.custom_heatmap_colors = settings["custom_heatmap_colors"]

            messagebox.showinfo("Success", "Settings loaded successfully.")

def main():
    root = tk.Tk()
    app = ClusteringGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
