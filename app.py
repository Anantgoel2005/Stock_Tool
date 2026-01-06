import logging
import threading
import tkinter as tk
from typing import Optional
from tkinter import messagebox, ttk
from tkinter.scrolledtext import ScrolledText

from stock_tool import (
    DEFAULT_HORIZON,
    DEFAULT_NEWS_LOOKBACK,
    DEFAULT_SYMBOL,
    get_live_prediction_with_reasoning,
    train_and_save_model,
)


class TextHandler(logging.Handler):
    """Logging handler that writes to a Tkinter text widget."""

    def __init__(self, text_widget: ScrolledText):
        super().__init__()
        self.text_widget = text_widget

    def emit(self, record):
        msg = self.format(record) + "\n"
        self.text_widget.after(0, self._append, msg)

    def _append(self, msg: str):
        self.text_widget.insert(tk.END, msg)
        self.text_widget.see(tk.END)


class StockApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Dynamic Stock Prediction Tool")
        self.geometry("1000x650")
        self.resizable(True, True)

        self.ticker_var = tk.StringVar(value=DEFAULT_SYMBOL)
        self.horizon_var = tk.StringVar(value=DEFAULT_HORIZON)
        self.news_days_var = tk.IntVar(value=DEFAULT_NEWS_LOOKBACK)
        self.news_query_var = tk.StringVar(value="")

        self._worker: Optional[threading.Thread] = None

        self._setup_logging()
        self._build_ui()

    def _setup_logging(self):
        self.log_area = ScrolledText(self, height=18, wrap=tk.WORD)
        self.log_area.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True, padx=10, pady=10)

        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        handler = TextHandler(self.log_area)
        handler.setFormatter(formatter)

        logging.basicConfig(level=logging.INFO, handlers=[handler])
        logging.getLogger("transformers").setLevel(logging.WARNING)

    def _build_ui(self):
        top_frame = ttk.Frame(self, padding=10)
        top_frame.pack(side=tk.TOP, fill=tk.X)

        ttk.Label(top_frame, text="Ticker:").grid(row=0, column=0, sticky=tk.W, pady=5)
        ticker_entry = ttk.Entry(top_frame, textvariable=self.ticker_var, width=20)
        ticker_entry.grid(row=0, column=1, sticky=tk.W, pady=5, padx=(0, 15))

        ttk.Label(top_frame, text="Investment Horizon:").grid(
            row=0, column=2, sticky=tk.W
        )
        horizon_combo = ttk.Combobox(
            top_frame,
            textvariable=self.horizon_var,
            values=("short_term", "long_term"),
            state="readonly",
            width=15,
        )
        horizon_combo.grid(row=0, column=3, sticky=tk.W, padx=(0, 15))

        ttk.Label(top_frame, text="News Lookback (days):").grid(
            row=0, column=4, sticky=tk.W
        )
        days_spin = ttk.Spinbox(
            top_frame,
            from_=1,
            to=30,
            textvariable=self.news_days_var,
            width=5,
        )
        days_spin.grid(row=0, column=5, sticky=tk.W)

        ttk.Label(top_frame, text="News Query (optional):").grid(
            row=1, column=0, sticky=tk.W, pady=5
        )
        news_entry = ttk.Entry(top_frame, textvariable=self.news_query_var, width=40)
        news_entry.grid(row=1, column=1, columnspan=2, sticky=tk.W, pady=5)

        button_frame = ttk.Frame(self, padding=10)
        button_frame.pack(fill=tk.X)

        train_button = ttk.Button(
            button_frame, text="Train / Retrain Model", command=self._handle_train
        )
        train_button.pack(side=tk.LEFT, padx=(0, 10))

        predict_button = ttk.Button(
            button_frame, text="Run Live Prediction", command=self._handle_predict
        )
        predict_button.pack(side=tk.LEFT, padx=(0, 10))

        self.status_var = tk.StringVar(value="Ready.")
        status_label = ttk.Label(button_frame, textvariable=self.status_var)
        status_label.pack(side=tk.LEFT, padx=10)

        self.output_area = ScrolledText(self, height=12, wrap=tk.WORD)
        self.output_area.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))
        self.output_area.insert(tk.END, "Prediction output will appear here.\n")
        self.output_area.configure(state=tk.DISABLED)

    def _handle_train(self):
        self._run_task(self._train_task, "Training model...")

    def _handle_predict(self):
        self._run_task(self._predict_task, "Running live prediction...")

    def _run_task(self, target, status_message: str):
        if self._worker and self._worker.is_alive():
            messagebox.showinfo(
                "Please wait", "Another task is currently running. Please wait."
            )
            return

        ticker = self.ticker_var.get().strip().upper()
        if not ticker:
            messagebox.showerror("Invalid input", "Please enter a stock ticker.")
            return

        self.status_var.set(status_message)
        self._worker = threading.Thread(target=target, args=(ticker,), daemon=True)
        self._worker.start()

    def _train_task(self, ticker: str):
        horizon = self.horizon_var.get()
        success = train_and_save_model(ticker, investment_horizon=horizon)
        self.after(0, self._complete_task, success, "Training complete.")

    def _predict_task(self, ticker: str):
        horizon = self.horizon_var.get()
        days_back = max(1, int(self.news_days_var.get() or 1))
        news_query = self.news_query_var.get().strip() or ticker
        result = get_live_prediction_with_reasoning(
            ticker,
            news_query=news_query,
            investment_horizon=horizon,
            days_back_for_news=days_back,
        )
        self.after(0, self._show_prediction, result)
        self.after(0, self._complete_task, True, "Prediction complete.")

    def _show_prediction(self, text: str):
        self.output_area.configure(state=tk.NORMAL)
        self.output_area.delete("1.0", tk.END)
        self.output_area.insert(tk.END, text)
        self.output_area.configure(state=tk.DISABLED)

    def _complete_task(self, success: bool, message: str):
        self.status_var.set(message if success else "Task failed. Check logs.")


if __name__ == "__main__":
    app = StockApp()
    app.mainloop()

