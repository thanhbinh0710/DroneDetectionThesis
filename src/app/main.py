# src/app/main.py
import sys
import os
import pyqtgraph as pg
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QLabel, QFrame)
from PyQt6.QtCore import Qt

# Import components
from .threads import DataWorker
from .ui.components import ResultPanel, SystemResultWidget

class DashboardApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Hệ thống Phát hiện Drone - Phân tích Âm thanh")
        self.resize(1000, 700)
        self.setup_ui()
        
        # Backend
        self.worker = DataWorker()
        self.worker.data_signal.connect(self.update_dashboard)
        self.worker.start()

    def setup_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        
        # Main layout
        main_layout = QVBoxLayout(main_widget)
        main_layout.setSpacing(15)
        main_layout.setContentsMargins(20, 20, 20, 20)
        
        # === TITLE SECTION ===
        title_label = QLabel("HỆ THỐNG PHÁT HIỆN DRONE BẰNG ÂM THANH")
        title_label.setObjectName("SectionTitle")
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_font = title_label.font()
        title_font.setPointSize(16)
        title_font.setBold(True)
        title_label.setFont(title_font)
        main_layout.addWidget(title_label)
        
        # === AUDIO RESULT PANEL ===
        results_container = QFrame()
        results_container.setObjectName("ResultsPanel")
        results_layout = QVBoxLayout(results_container)
        results_layout.setSpacing(10)
        results_layout.setContentsMargins(20, 20, 20, 20)
        
        # Audio detection result panel (centered)
        self.result_audio = ResultPanel("Phát hiện từ Âm thanh (Audio Detection)")
        results_layout.addWidget(self.result_audio, alignment=Qt.AlignmentFlag.AlignCenter)
        
        main_layout.addWidget(results_container)
        
        # === SYSTEM RESULT ===
        self.system_result = SystemResultWidget()
        main_layout.addWidget(self.system_result)
        
        # === INFO SECTION ===
        info_layout = QHBoxLayout()
        
        # Date/Time
        datetime_frame = QFrame()
        datetime_layout = QVBoxLayout(datetime_frame)
        lbl_datetime_title = QLabel("Thời gian phát hiện")
        self.lbl_datetime = QLabel("2024-01-01  00:00:00")
        self.lbl_datetime.setObjectName("DateTimeValue")
        datetime_layout.addWidget(lbl_datetime_title)
        datetime_layout.addWidget(self.lbl_datetime)
        
        # Audio Info
        audio_info_frame = QFrame()
        audio_info_layout = QVBoxLayout(audio_info_frame)
        lbl_audio_info_title = QLabel("Thông tin âm thanh")
        self.lbl_audio_info = QLabel("Tần số lấy mẫu: 44100 Hz")
        self.lbl_audio_info.setObjectName("PositionValue")
        audio_info_layout.addWidget(lbl_audio_info_title)
        audio_info_layout.addWidget(self.lbl_audio_info)
        
        # Detection count
        count_frame = QFrame()
        count_layout = QVBoxLayout(count_frame)
        lbl_count_title = QLabel("Tổng số lần phát hiện")
        self.lbl_detection_count = QLabel("0")
        self.lbl_detection_count.setObjectName("ElevValue")
        self.detection_count = 0
        count_layout.addWidget(lbl_count_title)
        count_layout.addWidget(self.lbl_detection_count)
        
        info_layout.addWidget(datetime_frame)
        info_layout.addWidget(audio_info_frame)
        info_layout.addWidget(count_frame)
        info_layout.addStretch()
        
        main_layout.addLayout(info_layout)
        
        # === CONFIDENCE HISTORY GRAPH ===
        graph_label = QLabel("Biểu đồ Độ tin cậy theo Thời gian")
        graph_label.setObjectName("SectionTitle")
        main_layout.addWidget(graph_label)
        
        self.graph = pg.PlotWidget()
        self.graph.setBackground('w')
        self.graph.showGrid(x=True, y=True, alpha=0.3)
        self.graph.setLabel('bottom', 'Thời gian (giây)', **{'color': '#333', 'font-size': '10pt'})
        self.graph.setLabel('left', 'Độ tin cậy', **{'color': '#333', 'font-size': '10pt'})
        self.graph.setYRange(0, 1.0)
        self.graph.setXRange(0, 60)
        
        # Confidence history data
        self.confidence_history = []
        self.time_history = []
        self.plot_line = self.graph.plot([], [], pen=pg.mkPen(color=(0, 120, 255), width=2))
        
        main_layout.addWidget(self.graph)

    def update_dashboard(self, prediction_data):
        """Update dashboard with new detection data from model predictions
        
        Args:
            prediction_data: Dict containing:
                - confidence: float (0-1)
                - status: str ('DRONE' or '-')
                - source: str ('model' or 'simulated')
                - file: str (audio filename)
        """
        from PyQt6.QtCore import QDateTime
        
        # Update datetime
        now = QDateTime.currentDateTime()
        self.lbl_datetime.setText(now.toString("yyyy-MM-dd  HH:mm:ss"))
        
        # Get prediction data
        confidence = prediction_data['confidence']
        status = prediction_data['status']
        source = prediction_data.get('source', 'unknown')
        audio_file = prediction_data.get('file', 'N/A')
        
        # Update audio info display
        info_text = f"Sample Rate: 44100 Hz"
        if source == 'model':
            info_text += f" | File: {audio_file} | Source: ML Model"
        else:
            info_text += f" | Source: Simulated"
        self.lbl_audio_info.setText(info_text)
        
        # Update audio result panel
        self.result_audio.update_result(status, confidence)
        
        # Update system result (same as audio since only one sensor)
        self.system_result.update_result(status, confidence)
        
        # Update detection count
        if status == "DRONE":
            self.detection_count += 1
            self.lbl_detection_count.setText(str(self.detection_count))
        
        # Update confidence history graph
        current_time = len(self.confidence_history)
        self.confidence_history.append(confidence)
        self.time_history.append(current_time)
        
        # Keep only last 60 data points
        if len(self.confidence_history) > 60:
            self.confidence_history.pop(0)
            self.time_history.pop(0)
            # Adjust time values
            self.time_history = [t - 1 for t in self.time_history]
        
        # Update plot
        self.plot_line.setData(self.time_history, self.confidence_history)

    def closeEvent(self, event):
        self.worker.stop()
        event.accept()

# --- LOAD STYLE HELPER ---
def load_stylesheet(app):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    style_path = os.path.join(current_dir, '..', '..', 'styles', 'style.qss')
    if os.path.exists(style_path):
        with open(style_path, "r", encoding="utf-8") as f:
            app.setStyleSheet(f.read())

if __name__ == "__main__":
    app = QApplication(sys.argv)
    load_stylesheet(app)
    window = DashboardApp()
    window.show()
    sys.exit(app.exec())
