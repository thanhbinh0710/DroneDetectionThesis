# src/app/ui/components.py
from PyQt6.QtWidgets import QFrame, QVBoxLayout, QLabel
from PyQt6.QtCore import Qt

# Result Panel Component for audio detection
class ResultPanel(QFrame):
    def __init__(self, title):
        super().__init__()
        self.setObjectName("ResultPanel")
        self.confidence = 0.0
        
        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        layout.setContentsMargins(15, 15, 15, 15)
        
        # Title
        lbl_title = QLabel(title)
        lbl_title.setObjectName("ResultTitle")
        layout.addWidget(lbl_title)
        
        # Status label (DRONE/-)
        self.lbl_status = QLabel("DRONE")
        self.lbl_status.setObjectName("StatusDrone")
        self.lbl_status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.lbl_status)
        
        # Confidence
        conf_layout = QVBoxLayout()
        lbl_conf_title = QLabel("Confidence")
        lbl_conf_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_confidence = QLabel("0.72")
        self.lbl_confidence.setObjectName("ConfidenceValue")
        self.lbl_confidence.setAlignment(Qt.AlignmentFlag.AlignCenter)
        conf_layout.addWidget(lbl_conf_title)
        conf_layout.addWidget(self.lbl_confidence)
        layout.addLayout(conf_layout)
        
    def update_result(self, status, confidence):
        """Update result panel with detection status and confidence"""
        self.confidence = confidence
        self.lbl_status.setText(status)
        self.lbl_confidence.setText(f"{confidence:.2f}")
        
        # Update style based on status
        if status == "DRONE":
            self.lbl_status.setObjectName("StatusDrone")
        else:
            self.lbl_status.setObjectName("StatusNone")
        self.lbl_status.style().unpolish(self.lbl_status)
        self.lbl_status.style().polish(self.lbl_status)


# System Result Widget - Overall Detection Result
class SystemResultWidget(QFrame):
    def __init__(self):
        super().__init__()
        self.setObjectName("SystemResultPanel")
        
        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        layout.setContentsMargins(20, 20, 20, 20)
        
        # Title
        lbl_title = QLabel("DETECTION RESULT")
        lbl_title.setObjectName("SystemResultTitle")
        lbl_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(lbl_title)
        
        # Status
        self.lbl_status = QLabel("DRONE")
        self.lbl_status.setObjectName("SystemStatusDrone")
        self.lbl_status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.lbl_status)
        
        # Confidence
        lbl_conf_title = QLabel("Confidence")
        lbl_conf_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl_conf_title.setObjectName("SystemConfTitle")
        
        self.lbl_confidence = QLabel("0.00")
        self.lbl_confidence.setObjectName("SystemConfValue")
        self.lbl_confidence.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        layout.addWidget(lbl_conf_title)
        layout.addWidget(self.lbl_confidence)
        
    def update_result(self, status, confidence):
        """Update system result with overall detection status"""
        self.lbl_status.setText(status)
        self.lbl_confidence.setText(f"{confidence:.2f}")
