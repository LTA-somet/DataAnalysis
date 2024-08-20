import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from fpdf import FPDF
from sklearn.preprocessing import MinMaxScaler

class DataProcessorApp:
    def __init__(self, root):
        self.root = root
        self.root.title('Data Processor')
        self.root.geometry('1200x800')

        # Initialize Notebook (tabs)
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        # Tab 1: Process Data
        self.process_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.process_frame, text='Process Data')

        # Tab 2: View Data
        self.view_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.view_frame, text='View Data')

        # Tab 3: Charts
        self.chart_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.chart_frame, text='Charts')

        # Control buttons on the Process Data tab
        self.load_button = tk.Button(self.process_frame, text='Load CSV File', command=self.load_file)
        self.load_button.pack(pady=10)

        self.process_button = tk.Button(self.process_frame, text='Process Data', command=self.process_data, state=tk.DISABLED)
        self.process_button.pack(pady=10)

        self.clean_button = tk.Button(self.process_frame, text='Clean Data', command=self.clean_data, state=tk.DISABLED)
        self.clean_button.pack(pady=10)

        self.normalize_button = tk.Button(self.process_frame, text='Normalize Data (Min-Max)', command=self.normalize_data, state=tk.DISABLED)
        self.normalize_button.pack(pady=10)

        self.save_button = tk.Button(self.process_frame, text='Save Processed Data', command=self.save_file, state=tk.DISABLED)
        self.save_button.pack(pady=10)

        # Control buttons on the View Data tab
        self.show_info_button = tk.Button(self.view_frame, text='Show Data Info', command=self.show_info, state=tk.DISABLED)
        self.show_info_button.pack(pady=10)

        self.show_stats_button = tk.Button(self.view_frame, text='Show Statistics', command=self.show_statistics, state=tk.DISABLED)
        self.show_stats_button.pack(pady=5)

        self.export_sorted_asc_button = tk.Button(self.view_frame, text='Export Sorted Data (Ascending)', command=self.export_sorted_data_ascending, state=tk.DISABLED)
        self.export_sorted_asc_button.pack(pady=5)

        self.export_sorted_desc_button = tk.Button(self.view_frame, text='Export Sorted Data (Descending)', command=self.export_sorted_data_descending, state=tk.DISABLED)
        self.export_sorted_desc_button.pack(pady=5)

        self.export_sorted_by_date_button = tk.Button(self.view_frame, text='Export Sorted by Date (Earliest to Latest)', command=self.export_sorted_by_date, state=tk.DISABLED)
        self.export_sorted_by_date_button.pack(pady=5)

        self.data_text = tk.Text(self.view_frame, wrap=tk.WORD, height=10, width=100)
        self.data_text.pack(fill=tk.BOTH, expand=True, pady=10)

        self.data_table = ttk.Treeview(self.view_frame, columns=[], show='headings')
        self.data_table.pack(fill=tk.BOTH, expand=True)

        # Control buttons on the Charts tab
        self.plot_button = tk.Button(self.chart_frame, text='Plot Salary Distribution', command=self.plot_salary_distribution, state=tk.DISABLED)
        self.plot_button.pack(pady=10)

        self.canvas = None
        self.df = None

    def load_file(self):
        file_path = filedialog.askopenfilename(filetypes=[('CSV Files', '*.csv')])
        if file_path:
            try:
                self.df = pd.read_csv(file_path)
                messagebox.showinfo("Info", "File loaded successfully!")
                self.process_button.config(state=tk.NORMAL)
                self.show_info_button.config(state=tk.NORMAL)
                self.show_stats_button.config(state=tk.NORMAL)
                self.export_sorted_asc_button.config(state=tk.NORMAL)
                self.export_sorted_desc_button.config(state=tk.NORMAL)
                self.export_sorted_by_date_button.config(state=tk.NORMAL)
                self.plot_button.config(state=tk.NORMAL)
                self.normalize_button.config(state=tk.NORMAL)
                self.update_table()
                self.display_data()
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load file: {e}")

    def display_data(self):
        df_to_display = self.df
        if df_to_display is not None:
            self.data_text.delete(1.0, tk.END)
            content = df_to_display.head(20).to_string(index=False)
            self.data_text.insert(tk.END, content)
        else:
            self.data_text.delete(1.0, tk.END)
            self.data_text.insert(tk.END, "No data to display")

    def process_data(self):
        if self.df is not None:
            try:
                self.df['Id'] = pd.to_datetime(self.df['Id'], errors='coerce')
                # self.df['JoinDate'] = pd.to_datetime(self.df['JoinDate'], errors='coerce')
                # self.df['Salary'] = self.df['Salary'].fillna(self.df['Salary'].mean())

                # Handle missing values for JoinDate
                # most_frequent_join_date = self.df['JoinDate'].mode().dropna().iloc[0]
                # self.df['JoinDate'] = self.df['JoinDate'].fillna(most_frequent_join_date)

                # Handle missing values for DateOfBirth
                most_frequent_dob = self.df['Id'].mode().dropna().iloc[0]
                self.df['Id'] = self.df['Id'].fillna(most_frequent_dob)

                # Optional: Parse dates (if needed)
                def parse_date(date):
                    try:
                        return pd.to_datetime(date, format='%Y-%m-%d', errors='coerce')
                    except:
                        return pd.to_datetime(date, errors='coerce')

                self.df['Id'] = self.df['Id'].apply(parse_date)

                messagebox.showinfo("Info", "Data processed successfully!")
                self.save_button.config(state=tk.NORMAL)
                self.clean_button.config(state=tk.NORMAL)
                self.update_table()
                self.display_data()
            except Exception as e:
                messagebox.showerror("Error", f"Data processing failed: {e}")
        else:
            messagebox.showwarning("Warning", "No file loaded!")

    def clean_data(self):
        if self.df is not None:
            try:
                # Remove duplicate rows based on 'Name' and 'DateOfBirth'
                self.df.drop_duplicates(subset=['fixed acidity', 'Id','alcohol'], keep='first', inplace=True)

                # Drop rows with any missing values
                self.df.dropna(inplace=True)

                # Drop columns that are completely empty
                self.df.dropna(axis=1, how='all', inplace=True)

                messagebox.showinfo("Info", "Data cleaned successfully!")
                self.update_table()
                self.display_data()
            except Exception as e:
                messagebox.showerror("Error", f"Data cleaning failed: {e}")
        else:
            messagebox.showwarning("Warning", "No data to clean!")

    def normalize_data(self):
        if self.df is not None:
            try:
                # Select numeric columns for normalization, excluding 'ID'
                numeric_columns = self.df.select_dtypes(include=['number']).columns
                if 'ID' in numeric_columns:
                    numeric_columns = numeric_columns.drop('ID')
                scaler = MinMaxScaler()

                self.df[numeric_columns] = scaler.fit_transform(self.df[numeric_columns])
                messagebox.showinfo("Info", "Data normalized successfully!")
                self.update_table()
                self.display_data()
            except Exception as e:
                messagebox.showerror("Error", f"Data normalization failed: {e}")
        else:
            messagebox.showwarning("Warning", "No data to normalize!")

    def save_file(self):
        if self.df is not None:
            file_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[('CSV Files', '*.csv')])
            if file_path:
                try:
                    self.df.to_csv(file_path, index=False)
                    messagebox.showinfo("Info", "File saved successfully!")
                    self.save_report()
                except Exception as e:
                    messagebox.showerror("Error", f"Failed to save file: {e}")
        else:
            messagebox.showwarning("Warning", "No data to save!")

    def export_sorted_data_ascending(self):
        if self.df is not None:
            try:
                sorted_df = self.df.sort_values(by=['Salary'], ascending=True)
                file_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[('CSV Files', '*.csv')])
                if file_path:
                    sorted_df.to_csv(file_path, index=False)
                    messagebox.showinfo("Info", "Sorted data (ascending) exported successfully!")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to export sorted data (ascending): {e}")
        else:
            messagebox.showwarning("Warning", "No data to export!")

    def export_sorted_data_descending(self):
        if self.df is not None:
            try:
                sorted_df = self.df.sort_values(by=['Salary'], ascending=False)
                file_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[('CSV Files', '*.csv')])
                if file_path:
                    sorted_df.to_csv(file_path, index=False)
                    messagebox.showinfo("Info", "Sorted data (descending) exported successfully!")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to export sorted data (descending): {e}")
        else:
            messagebox.showwarning("Warning", "No data to export!")

    def export_sorted_by_date(self):
        if self.df is not None:
            try:
                sorted_df = self.df.sort_values(by=['JoinDate'], ascending=True)
                sorted_df.drop_duplicates(inplace=True)  # Remove duplicate rows
                file_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[('CSV Files', '*.csv')])
                if file_path:
                    sorted_df.to_csv(file_path, index=False)
                    messagebox.showinfo("Info", "Sorted data by date exported successfully!")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to export sorted data by date: {e}")
        else:
            messagebox.showwarning("Warning", "No data to export!")

    def save_report(self):
        try:
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", size=12)
            pdf.cell(200, 10, txt="Data Summary Report", ln=True, align='C')
            pdf.ln(10)

            summary = self.df.describe(include='all').to_string()
            pdf.multi_cell(0, 10, summary)

            file_path = filedialog.asksaveasfilename(defaultextension=".pdf", filetypes=[('PDF Files', '*.pdf')])
            if file_path:
                pdf.output(file_path)
                messagebox.showinfo("Info", "Report saved successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save report: {e}")

    def show_info(self):
        if self.df is not None:
            num_rows = len(self.df)
            num_cols = len(self.df.columns)
            col_names = ', '.join(self.df.columns)
            dtypes = self.df.dtypes
            missing_values = self.df.isnull().sum()
            unique_values = self.df.nunique()

            info = (
                f"Number of rows: {num_rows}\n"
                f"Number of columns: {num_cols}\n"
                f"Column names: {col_names}\n"
                f"Data types:\n{dtypes}\n\n"
                f"Missing values:\n{missing_values}\n\n"
                f"Unique values:\n{unique_values}"
            )
            messagebox.showinfo("Data Info", info)
        else:
            messagebox.showwarning("Warning", "No data to show!")

    def show_statistics(self):
        if self.df is not None:
            try:
                if 'Salary' in self.df.columns:
                    salary_stats = self.df['Salary'].describe().to_string()
                    messagebox.showinfo("Salary Statistics", f"Salary Statistics:\n\n{salary_stats}")
                else:
                    messagebox.showwarning("Warning", "The 'Salary' column is not available in the data.")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to retrieve salary statistics: {e}")
        else:
            messagebox.showwarning("Warning", "No data available to show salary statistics!")

    def update_table(self):
        if self.df is not None:
            self.data_table["columns"] = list(self.df.columns)
            for col in self.df.columns:
                self.data_table.heading(col, text=col)

            self.data_table.delete(*self.data_table.get_children())

            for _, row in self.df.iterrows():
                self.data_table.insert("", "end", values=list(row))

    def show_salary_info(self):
        if self.df is not None:
            try:
                salary_info = self.df['Salary'].describe()
                info_text = (
                    f"Salary Information:\n"
                    f"Count: {salary_info['count']}\n"
                    f"Mean: {salary_info['mean']}\n"
                    f"Standard Deviation: {salary_info['std']}\n"
                    f"Min: {salary_info['min']}\n"
                    f"25th Percentile: {salary_info['25%']}\n"
                    f"Median: {salary_info['50%']}\n"
                    f"75th Percentile: {salary_info['75%']}\n"
                    f"Max: {salary_info['max']}"
                )
                messagebox.showinfo("Salary Info", info_text)
            except Exception as e:
                messagebox.showerror("Error", f"Failed to retrieve salary info: {e}")
        else:
            messagebox.showwarning("Warning", "No data available to show salary info!")

    def plot_salary_distribution(self):
        if self.df is not None:
            try:
                plt.figure(figsize=(8, 6))
                self.df['Salary'].hist(bins=20)
                plt.title('Salary Distribution')
                plt.xlabel('Salary')
                plt.ylabel('Frequency')

                if self.canvas:
                    self.canvas.get_tk_widget().destroy()

                self.canvas = FigureCanvasTkAgg(plt.gcf(), master=self.chart_frame)
                self.canvas.draw()
                self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            except Exception as e:
                messagebox.showerror("Error", f"Failed to plot salary distribution: {e}")
        else:
            messagebox.showwarning("Warning", "No data to plot!")

# Create the GUI
root = tk.Tk()
app = DataProcessorApp(root)
root.mainloop()
