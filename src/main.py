import logging
import threading
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext

import config
from abstract_topic_modeling import abstract_topic_modeling_pipeline
from clean_data import clean_df_pipeline
from knowledge_graph import knowledge_graph_pipeline, open_graph


# --- Custom Logging Handler for Tkinter Text Widget ---
class TextWidgetHandler(logging.Handler):
    """
    A custom logging handler that sends log records to a Tkinter Text widget.
    """

    def __init__(self, text_widget):
        super().__init__()
        self.text_widget = text_widget
        self.text_widget.config(state='disabled')  # read-only

    def emit(self, record):
        """
        Emits a log record to the text widget.
        """
        msg = self.format(record)
        self.text_widget.after(0, self._insert_text, msg + "\n")

    def _insert_text(self, text):
        """Helper to insert text into the widget from the main thread."""
        self.text_widget.config(state='normal')
        self.text_widget.insert(tk.END, text)
        self.text_widget.see(tk.END)  # Scroll to the end
        self.text_widget.config(state='disabled')


# --- Main execution function for GUI ---
def generate_session_topics_and_knowledge_graph_in_gui(log_text_widget, status_label, root_window, execute_button,
                                                       force_override_var, original_graph_button,
                                                       generated_graph_button):
    # Disable buttons to prevent multiple clicks
    execute_button.config(state='disabled')
    original_graph_button.config(state='disabled')
    generated_graph_button.config(state='disabled')

    status_label.config(text="Status: Generating session topic suggestions...")
    log_text_widget.config(state='normal')
    log_text_widget.delete(1.0, tk.END)
    log_text_widget.config(state='disabled')

    # Get checkbox
    force_override = force_override_var.get()

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)  # Set the minimum level to capture
    # Remove any existing handlers to prevent duplicate output
    for handler in list(root_logger.handlers):
        # Only remove TextWidgetHandler instances to avoid removing other useful handlers
        if isinstance(handler, TextWidgetHandler):
            root_logger.removeHandler(handler)

    # Add custom handler
    text_handler = TextWidgetHandler(log_text_widget)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%H:%M:%S')
    text_handler.setFormatter(formatter)
    root_logger.addHandler(text_handler)

    def run_pipeline_target():
        try:
            if force_override:
                root_logger.info("Generating session topics (forced).")
                abstract_topic_modeling_pipeline(force_override=True)
            else:
                root_logger.info("Generating session topics, if they do not exist yet")
                abstract_topic_modeling_pipeline()

            root_logger.info("Session topic generation completed. Now generating knowledge graph.")
            knowledge_graph_pipeline()

            root_logger.info(f"Original data graph saved to: {config.GRAPH_ORIGINAL_DATA_PATH}")
            root_logger.info(f"Generated data graph saved to: {config.GRAPH_GENERATED_DATA_PATH}")
            root_logger.info("You can now open the graph files using the buttons.")  # Updated message

            # Schedule messagebox and status updates on the main thread
            root_window.after(0, lambda: messagebox.showinfo("Execution Result",
                                                             "Session topic suggestions and knowledge graph generated successfully!\n\nYou can now open the graph files using the buttons."))
            # Enable the graph opening buttons on the main thread
            root_window.after(0, lambda: original_graph_button.config(state='normal'))
            root_window.after(0, lambda: generated_graph_button.config(state='normal'))

        except Exception as e:
            # Log the error through the logger, which will go to the text widget
            root_logger.error(f"Pipeline failed: {e}", exc_info=True)  # Generalizing error message
            # Schedule an error message box on the main thread
            root_window.after(0, lambda: messagebox.showerror("Error", f"Pipeline failed: {e}"))
        finally:
            root_window.after(0, lambda: status_label.config(text="Status: Complete!"))
            root_window.after(0, lambda: execute_button.config(state='normal'))

            # Remove the handler from the logger when done
            root_logger.removeHandler(text_handler)

    # Create and start the new thread
    thread = threading.Thread(target=run_pipeline_target)
    thread.daemon = True  # Allow the main program to exit even if thread is still running
    thread.start()


# --- GUI Setup ---
def run():
    root = tk.Tk()
    root.title("Application")
    root.geometry("600x600")
    root.resizable(True, True)

    style = ttk.Style(root)
    try:
        style.theme_use('clam')
    except tk.TclError:
        print("Clam theme not found, using default.")
        style.theme_use('default')

    main_frame = ttk.Frame(root, padding="20 20 20 20")
    main_frame.pack(expand=True, fill='both')

    label = ttk.Label(main_frame, text="IMPORTANT: Please ensure Gephy (https://gephi.org/) is installed.",
                      font=("Arial", 10))
    label.pack(pady=10)

    execute_button = ttk.Button(main_frame, text="Generate topic suggestions and knowledge graph file")
    execute_button.pack(pady=10)

    # Variable to hold the state of the checkbox
    force_override_var = tk.BooleanVar(value=False)
    # Checkbox for force_override
    override_checkbox = ttk.Checkbutton(main_frame,
                                        text="Force Override Existing Data",
                                        variable=force_override_var,
                                        onvalue=True, offvalue=False)
    override_checkbox.pack(pady=5)

    # --- Graph Opening Buttons Frame ---
    graph_buttons_frame = ttk.Frame(main_frame)
    graph_buttons_frame.pack(pady=5)

    original_graph_button = ttk.Button(graph_buttons_frame, text="Open Original Data Graph",
                                       command=lambda: open_graph(config.GRAPH_ORIGINAL_DATA_PATH))
    original_graph_button.pack(side='left', padx=5)
    original_graph_button.config(state='disabled')

    generated_graph_button = ttk.Button(graph_buttons_frame, text="Open Generated Data Graph",
                                        command=lambda: open_graph(config.GRAPH_GENERATED_DATA_PATH))
    generated_graph_button.pack(side='left', padx=5)
    generated_graph_button.config(state='disabled')
    # --- End Graph Opening Buttons ---

    status_label = ttk.Label(main_frame, text="Status: Ready", font=("Arial", 9), foreground="gray")
    status_label.pack(pady=5)

    log_label = ttk.Label(main_frame, text="Program Log:", font=("Arial", 10))
    log_label.pack(pady=(10, 5), anchor='w')

    log_text_widget = scrolledtext.ScrolledText(main_frame, wrap=tk.WORD, width=60, height=15,
                                                font=("Consolas", 9), state='disabled',
                                                borderwidth=1, relief="sunken")
    log_text_widget.pack(expand=True, fill='both', padx=5, pady=5)

    execute_button.config(
        command=lambda: generate_session_topics_and_knowledge_graph_in_gui(
            log_text_widget, status_label, root, execute_button,
            force_override_var, original_graph_button, generated_graph_button
        )
    )

    try:
        clean_df_pipeline()
    except Exception as e:
        messagebox.showerror("Startup Error", f"An error occurred during initial data pipeline execution: {e}")

    root.mainloop()


if __name__ == "__main__":
    run()
