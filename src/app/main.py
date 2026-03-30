# src/app/main.py
import sys
import os
import pyqtgraph as pg
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QLabel, QFrame)
from PyQt6.QtCore import Qt

# Import components
from .threads import DataWorker

class DashboardApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Drone Detection System - Audio Analysis")
        self.resize(900, 600)
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
        
        # === HEADER SECTION ===
        header_layout = QHBoxLayout()
        
        title_label = QLabel("Drone Detection Dashboard")
        title_label.setObjectName("HeaderTitle")
        subtitle_label = QLabel("Drone Detection System using Audio Analysis")
        subtitle_label.setObjectName("HeaderSubtitle")
        
        title_vbox = QVBoxLayout()
        title_vbox.addWidget(title_label)
        title_vbox.addWidget(subtitle_label)
        title_vbox.setSpacing(0)
        
        header_layout.addLayout(title_vbox)
        header_layout.addStretch()
        main_layout.addLayout(header_layout)
        
        # === SEPARATOR LINE ===
        separator = QFrame()
        separator.setObjectName("SeparatorLine")
        separator.setFixedHeight(2)
        main_layout.addWidget(separator)
        
        # === INFO CARDS SECTION (4 cards) ===
        cards_layout = QHBoxLayout()
        cards_layout.setSpacing(15)
        
        
        
        # Card 1: Time
        time_card = self._create_info_card("TIME", "00:00:00")
        self.lbl_time_display = time_card[1]
        cards_layout.addWidget(time_card[0])
        
        # Card 2: Date
        date_card = self._create_info_card("DATE", "Monday, March 30, 2026")
        self.lbl_date_display = date_card[1]
        cards_layout.addWidget(date_card[0])
        
        # Card 3: Sample Rate
        sample_card = self._create_info_card("AUDIO SAMPLE RATE", "44100 Hz")
        self.lbl_sample_rate = sample_card[1]
        cards_layout.addWidget(sample_card[0])
        
        # Card 4: Detection Count
        count_card = self._create_info_card("DETECTION COUNT", "0")
        self.lbl_detection_count = count_card[1]
        self.detection_count = 0
        cards_layout.addWidget(count_card[0])
        
        main_layout.addLayout(cards_layout)
        
        # === MAIN CONTENT: 2 COLUMNS ===
        content_layout = QHBoxLayout()
        content_layout.setSpacing(15)
        
        # LEFT COLUMN: Status and Confidence Panels
        left_column_widget = QFrame()
        left_column_widget.setMaximumWidth(280)
        left_column_layout = QVBoxLayout(left_column_widget)
        left_column_layout.setSpacing(12)
        
        # Panel 1: Status
        status_panel = QFrame()
        status_panel.setObjectName("StatusPanel")
        status_layout = QVBoxLayout(status_panel)
        status_layout.setContentsMargins(12, 12, 12, 12)
        status_layout.setSpacing(8)
        
        lbl_status_title = QLabel("System Results")
        lbl_status_title.setObjectName("PanelTitle")
        status_layout.addWidget(lbl_status_title)
        
        # Status icon and text
        self.lbl_status = QLabel("DRONE")
        self.lbl_status.setObjectName("StatusDrone")
        self.lbl_status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        status_layout.addWidget(self.lbl_status)
        
        # Panel 2: Confidence
        confidence_panel = QFrame()
        confidence_panel.setObjectName("ConfidencePanel")
        confidence_layout = QVBoxLayout(confidence_panel)
        confidence_layout.setContentsMargins(12, 12, 12, 12)
        confidence_layout.setSpacing(8)
        
        # Confidence
        conf_title_label = QLabel("Confidence")
        conf_title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        conf_title_label.setObjectName("PanelTitle")
        self.lbl_confidence = QLabel("0.00")
        self.lbl_confidence.setObjectName("ConfidenceValue")
        self.lbl_confidence.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        confidence_layout.addWidget(conf_title_label)
        confidence_layout.addWidget(self.lbl_confidence)
        
        # Add both panels to left column
        left_column_layout.addWidget(status_panel, 1)
        left_column_layout.addWidget(confidence_panel, 1)
        
        content_layout.addWidget(left_column_widget, 1)
        
        # RIGHT COLUMN: Confidence History Graph
        right_panel = QFrame()
        right_panel.setObjectName("GraphPanel")
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(15, 15, 15, 15)
        right_layout.setSpacing(10)
        
        self.graph = pg.PlotWidget()
        self.graph.setBackground('#ffffff')
        self.graph.showGrid(x=True, y=True, alpha=0.2)
        self.graph.setLabel('bottom', 'Time (seconds)', **{'color': '#4a5568', 'font-size': '10pt'})
        self.graph.setLabel('left', 'Confidence (%)', **{'color': '#4a5568', 'font-size': '10pt'})
        self.graph.setYRange(0, 100)
        self.graph.setXRange(0, 60)
        
        # Confidence history data
        self.confidence_history = []
        self.time_history = []
        self.plot_line = self.graph.plot([], [], pen=pg.mkPen(color=(43, 165, 132), width=2))
        
        right_layout.addWidget(self.graph, 1)
        right_layout.addWidget(QLabel("LIVE DATA"), 0)
        
        content_layout.addWidget(right_panel, 2)
        
        main_layout.addLayout(content_layout, 1)
    
    def _create_info_card(self, title, value):
        """Create an info card with title and value"""
        card = QFrame()
        card.setObjectName("InfoCard")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(5)
        
        lbl_title = QLabel(title)
        lbl_title.setObjectName("CardTitle")
        
        lbl_value = QLabel(value)
        lbl_value.setObjectName("CardValue")
        
        layout.addWidget(lbl_title)
        layout.addWidget(lbl_value)
        
        return card, lbl_value

    def update_dashboard(self, prediction_data):
        """Update dashboard with new detection data from model predictions"""
        from PyQt6.QtCore import QDateTime, QTime
        
        # Update time
        now = QDateTime.currentDateTime()
        self.lbl_time_display.setText(now.toString("HH:mm:ss"))
        self.lbl_date_display.setText(now.toString("dddd, d MMMM yyyy"))
        
        # Get prediction data
        confidence = prediction_data['confidence']
        status = prediction_data['status']
        
        # Update status label
        self.lbl_status.setText(status)
        if status == "DRONE":
            self.lbl_status.setObjectName("StatusDrone")
        else:
            self.lbl_status.setObjectName("StatusNone")
        self.lbl_status.style().unpolish(self.lbl_status)
        self.lbl_status.style().polish(self.lbl_status)
        
        # Update confidence (display as decimal 0.00-1.00)
        self.lbl_confidence.setText(f"{confidence:.2f}")
        
        # Update detection count
        if status == "DRONE":
            self.detection_count += 1
            self.lbl_detection_count.setText(str(self.detection_count))
        
        # Update confidence history graph (use decimal 0.00-1.00)
        current_time = len(self.confidence_history)
        self.confidence_history.append(confidence)
        self.time_history.append(current_time)
        
        # Keep only last 60 data points
        if len(self.confidence_history) > 60:
            self.confidence_history.pop(0)
            self.time_history.pop(0)
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
