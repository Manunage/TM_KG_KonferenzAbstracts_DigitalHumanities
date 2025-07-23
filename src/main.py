import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import logging
import threading

from clean_data import clean_df_pipeline
from abstract_topic_modeling import abstract_topic_modeling_pipeline


# --- Custom Logging Handler for Tkinter Text Widget ---
class TextWidgetHandler(logging.Handler):
    """
    A custom logging handler that sends log records to a Tkinter Text widget.
    """
    def __init__(self, text_widget):
        super().__init__()
        self.text_widget = text_widget
        self.text_widget.config(state='disabled') # Ensure it's read-only by default

    def emit(self, record):
        """
        Emits a log record to the text widget.
        """
        msg = self.format(record)
        # Tkinter operations must be on the main thread.
        # Use after() to schedule the update on the main thread.
        self.text_widget.after(0, self._insert_text, msg + "\n")

    def _insert_text(self, text):
        """Helper to insert text into the widget from the main thread."""
        self.text_widget.config(state='normal')
        self.text_widget.insert(tk.END, text)
        self.text_widget.see(tk.END) # Scroll to the end
        self.text_widget.config(state='disabled')


# --- Main execution function for GUI ---
def execute_topic_modeling_pipeline_in_gui(log_text_widget, status_label, root_window, execute_button):
    """
    This function sets up the logging and starts the topic modeling pipeline
    in a separate thread to keep the GUI responsive.
    It also manages the state of the execute_button.
    """
    # Disable the button immediately to prevent multiple clicks
    execute_button.config(state='disabled')
    status_label.config(text="Status: Starting Topic Modeling in background...")
    log_text_widget.config(state='normal') # Enable for clearing
    log_text_widget.delete(1.0, tk.END) # Clear previous log
    log_text_widget.config(state='disabled') # Disable after clearing

    # Get the root logger and configure it to use our custom handler
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO) # Set the minimum level to capture
    # Remove any existing handlers to prevent duplicate output
    for handler in list(root_logger.handlers):
        # Only remove TextWidgetHandler instances to avoid removing other useful handlers
        if isinstance(handler, TextWidgetHandler):
            root_logger.removeHandler(handler)

    # Add our custom handler
    text_handler = TextWidgetHandler(log_text_widget)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%H:%M:%S')
    text_handler.setFormatter(formatter)
    root_logger.addHandler(text_handler)

    def run_pipeline_target():
        """
        This function will be executed in a separate thread.
        It contains the long-running task and its error handling.
        """
        try:
            abstract_topic_modeling_pipeline()
            # Use after() to schedule messagebox and status updates on the main thread
            root_window.after(0, lambda: messagebox.showinfo("Execution Result", "Abstract topic modeling pipeline completed successfully!"))
        except Exception as e:
            # Log the error through the logger, which will go to the text widget
            root_logger.error(f"Abstract topic modeling pipeline failed: {e}", exc_info=True)
            # Schedule an error message box on the main thread
            root_window.after(0, lambda: messagebox.showerror("Error", f"Abstract topic modeling pipeline failed: {e}"))
        finally:
            # Schedule final status update and re-enable button on the main thread
            root_window.after(0, lambda: status_label.config(text="Status: Complete!"))
            root_window.after(0, lambda: execute_button.config(state='normal')) # Re-enable the button
            # Remove the handler from the logger when done
            root_logger.removeHandler(text_handler)

    # Create and start the new thread
    thread = threading.Thread(target=run_pipeline_target)
    thread.daemon = True # Allow the main program to exit even if thread is still running
    thread.start()


# --- GUI Setup ---
def run():
    """
    Creates and displays the main GUI window for the application.
    """
    root = tk.Tk()
    root.title("Application")
    root.geometry("600x450")
    root.resizable(True, True)

    style = ttk.Style(root)
    try:
        style.theme_use('clam')
    except tk.TclError:
        print("Clam theme not found, using default.")
        style.theme_use('default')

    main_frame = ttk.Frame(root, padding="20 20 20 20")
    main_frame.pack(expand=True, fill='both')

    label = ttk.Label(main_frame, text="Click to execute the abstract topic modeling pipeline and view logs.",
                      font=("Arial", 10))
    label.pack(pady=10)

    execute_button = ttk.Button(main_frame, text="Execute Topic Modeling Pipeline")
    execute_button.pack(pady=10)

    status_label = ttk.Label(main_frame, text="Status: Ready", font=("Arial", 9), foreground="gray")
    status_label.pack(pady=5)

    log_label = ttk.Label(main_frame, text="Program Log:", font=("Arial", 10))
    log_label.pack(pady=(10, 5), anchor='w')

    log_text_widget = scrolledtext.ScrolledText(main_frame, wrap=tk.WORD, width=60, height=15,
                                                font=("Consolas", 9), state='disabled',
                                                borderwidth=1, relief="sunken")
    log_text_widget.pack(expand=True, fill='both', padx=5, pady=5)

    # Configure the button command to call the new execution function for topic modeling
    # Pass the root window reference and the button itself for thread-safe GUI updates
    execute_button.config(command=lambda: execute_topic_modeling_pipeline_in_gui(log_text_widget, status_label, root, execute_button))

    try:
        clean_df_pipeline()
    except Exception as e:
        messagebox.showerror("Startup Error", f"An error occurred during initial data pipeline execution: {e}")

    root.mainloop()


if __name__ == "__main__":
    run()
