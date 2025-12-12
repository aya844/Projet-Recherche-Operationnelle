
import sys
import csv
import math

# Try importing the solver module
try:
    import solveur
except ImportError:
    solveur = None 

from PyQt6.QtWidgets import (QApplication, QMainWindow, QGraphicsScene, QGraphicsView, 
                             QGraphicsEllipseItem, QGraphicsLineItem, QGraphicsItem, 
                             QToolBar, QVBoxLayout, QWidget, QInputDialog, QMessageBox,
                             QDialog, QFormLayout, QSpinBox, QDoubleSpinBox, QDialogButtonBox,
                             QTableWidget, QTableWidgetItem, QHeaderView, QFileDialog, QComboBox,
                             QGraphicsDropShadowEffect)
from PyQt6.QtCore import Qt, QPointF, QLineF, QRectF, QSize
from PyQt6.QtGui import (QPen, QBrush, QColor, QPainter, QFont, QRadialGradient, 
                         QLinearGradient, QPalette, QAction, QPainterPath, QIcon)

# --- Theme Constants ---
THEME_BG = QColor("#1e1e2e")        # Dark Slate Background
THEME_GRID = QColor("#2a2a3c")      # Slightly lighter grid
THEME_ACCENT = QColor("#3a86ff")    # Bright Blue Accent
THEME_TEXT = QColor("#ffffff")

CITY_COLOR_GRAD_1 = QColor("#4361ee")
CITY_COLOR_GRAD_2 = QColor("#3f37c9")
CITY_SELECTED = QColor("#f72585")

LINK_BASE_COLOR = QColor("#8d99ae")
LINK_AUGMENTED_COLOR = QColor("#06d6a0") # Green for added capacity
LINK_WIDTH_BASE = 3
LINK_WIDTH_SELECTED = 5

DEMAND_COLOR = QColor("#ef233c")

class CityItem(QGraphicsEllipseItem):
    def __init__(self, x, y, name, radius=22):
        super().__init__(-radius, -radius, 2*radius, 2*radius)
        self.setPos(x, y)
        self.name = name
        self.radius = radius
        
        self.setFlags(QGraphicsItem.GraphicsItemFlag.ItemIsMovable | 
                      QGraphicsItem.GraphicsItemFlag.ItemIsSelectable |
                      QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges)
        self.links = []
        
        # Drop Shadow for depth
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(15)
        shadow.setColor(QColor(0, 0, 0, 150))
        shadow.setOffset(3, 3)
        self.setGraphicsEffect(shadow)
        
        self.setAcceptHoverEvents(True)

    def itemChange(self, change, value):
        if change == QGraphicsItem.GraphicsItemChange.ItemPositionChange:
            for link in self.links:
                link.update_position()
        return super().itemChange(change, value)

    def paint(self, painter, option, widget):
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Gradient Fill
        grad = QRadialGradient(0, 0, self.radius)
        if self.isSelected():
            grad.setColorAt(0, CITY_SELECTED.lighter(120))
            grad.setColorAt(1, CITY_SELECTED)
        else:
            grad.setColorAt(0, CITY_COLOR_GRAD_1)
            grad.setColorAt(1, CITY_COLOR_GRAD_2)
            
        painter.setBrush(QBrush(grad))
        
        # Border
        if self.isSelected():
            painter.setPen(QPen(Qt.GlobalColor.white, 2.5))
        else:
            painter.setPen(Qt.PenStyle.NoPen)
            
        painter.drawEllipse(self.boundingRect())
        
        # Text Label (Name)
        painter.setPen(Qt.GlobalColor.white)
        font = QFont("Segoe UI", 9, QFont.Weight.Bold)
        painter.setFont(font)
        painter.drawText(self.boundingRect(), Qt.AlignmentFlag.AlignCenter, self.name)
        
class LinkItem(QGraphicsLineItem):
    def __init__(self, city1, city2, capacity, cost):
        super().__init__()
        self.city1 = city1
        self.city2 = city2
        self.base_capacity = capacity
        self.extra_capacity = 0.0
        self.cost = cost
        
        self.setPen(QPen(LINK_BASE_COLOR, LINK_WIDTH_BASE))
        self.setFlags(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable)
        
        self.city1.links.append(self)
        self.city2.links.append(self)
        self.update_position()
        self.setZValue(-1) # Send to back

    @property
    def capacity(self):
        return self.base_capacity + self.extra_capacity

    def update_position(self):
        line = QLineF(self.city1.pos(), self.city2.pos())
        self.setLine(line)
        
    def reset_capacity(self):
        self.extra_capacity = 0.0
        self.update()

    def add_extra_capacity(self, amount):
        self.extra_capacity = amount
        self.update()

    def paint(self, painter, option, widget):
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        is_augmented = self.extra_capacity > 1e-6
        
        # Determine Color & Width
        color = LINK_AUGMENTED_COLOR if is_augmented else LINK_BASE_COLOR
        width = LINK_WIDTH_BASE
        
        if self.isSelected():
            color = THEME_ACCENT
            width = LINK_WIDTH_SELECTED
        
        pen = QPen(color, width)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(pen)
        painter.drawLine(self.line())
        
        # --- Info Label ---
        mid = (self.line().p1() + self.line().p2()) / 2
        
        # Text Content
        text = f"{self.base_capacity:.0f}G"
        if is_augmented:
            text += f" + {self.extra_capacity:.0f}G"
        text += f" | {self.cost}DT"
        
        # Font Setup
        font = QFont("Segoe UI", 8, QFont.Weight.Bold)
        painter.setFont(font)
        fm = painter.fontMetrics()
        w = fm.horizontalAdvance(text) + 12
        h = fm.height() + 6
        
        # Label Rect
        rect = QRectF(mid.x() - w/2, mid.y() - h/2, w, h)
        
        # Label Background
        bg_col = QColor(20, 20, 20, 200)
        if is_augmented:
             bg_col = QColor(0, 80, 40, 220) # Dark Green bg for augmented
             
        path = QPainterPath()
        path.addRoundedRect(rect, 4, 4)
        painter.fillPath(path, bg_col)
        
        # Draw Text
        painter.setPen(QColor("#ffffff"))
        painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, text)

class DemandItem(QGraphicsLineItem):
    def __init__(self, city1, city2, traffic):
        super().__init__()
        self.city1 = city1
        self.city2 = city2
        self.traffic = traffic
        
        pen = QPen(DEMAND_COLOR, 2)
        pen.setStyle(Qt.PenStyle.DashLine)
        pen.setDashPattern([4, 4])
        self.setPen(pen)
        self.setFlags(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable)
        
        self.city1.links.append(self)
        self.city2.links.append(self)
        self.update_position()

    def update_position(self):
        self.setLine(QLineF(self.city1.pos(), self.city2.pos()))

    def paint(self, painter, option, widget):
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        p1 = self.city1.pos()
        p2 = self.city2.pos()
        
        painter.setPen(self.pen())
        painter.drawLine(p1, p2)

        # Draw Label below the line center
        mid = (p1 + p2) / 2
        label_pos = mid + QPointF(0, 20)
        
        text = f"Req: {self.traffic}G"
        font = QFont("Segoe UI", 8, QFont.Weight.Bold)
        painter.setFont(font)
        
        fm = painter.fontMetrics()
        w = fm.horizontalAdvance(text) + 8
        h = fm.height() + 4
        rect = QRectF(label_pos.x() - w/2, label_pos.y() - h/2, w, h)
        
        # Small background for readability
        painter.setBrush(QBrush(QColor(40, 10, 10, 200)))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRoundedRect(rect, 3, 3)
        
        painter.setPen(DEMAND_COLOR.lighter(130))
        painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, text)

class NetworkScene(QGraphicsScene):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.mode = "Select"
        self.start_city = None
        self.city_counter = 1
        self.setSceneRect(0, 0, 3000, 3000) # Large Canvas
        self.setBackgroundBrush(QBrush(THEME_BG))

    def drawBackground(self, painter, rect):
        super().drawBackground(painter, rect)
        
        # Subtle Grid System
        grid_step = 60
        left = int(rect.left()) - (int(rect.left()) % grid_step)
        top = int(rect.top()) - (int(rect.top()) % grid_step)
        
        # Dots instead of lines for a cleaner modern look
        painter.setPen(QPen(THEME_GRID, 1)) # Grid lines
        
        # Vertical lines
        for x in range(left, int(rect.right()), grid_step):
            painter.drawLine(x, int(rect.top()), x, int(rect.bottom()))
            
        # Horizontal lines
        for y in range(top, int(rect.bottom()), grid_step):
            painter.drawLine(int(rect.left()), y, int(rect.right()), y)

    def mousePressEvent(self, event):
        if self.mode == "AddCity":
            pos = event.scenePos()
            name, ok = QInputDialog.getText(None, "New City", "City Name:", text=f"City {self.city_counter}")
            if ok and name:
                city = CityItem(pos.x(), pos.y(), name)
                self.addItem(city)
                self.city_counter += 1
        
        elif self.mode == "AddLink":
            item = self.itemAt(event.scenePos(), QGraphicsView().transform())
            if isinstance(item, CityItem):
                if self.start_city is None:
                    self.start_city = item
                    item.setSelected(True)
                elif self.start_city == item:
                    item.setSelected(False)
                    self.start_city = None
                else:
                    # Create Link
                    end_city = item
                    dialog = LinkDialog()
                    if dialog.exec():
                        cap, cost = dialog.get_data()
                        LinkItem(self.start_city, end_city, cap, cost) # Adds itself to scene via items parent in simple impl? No, must add to scene.
                        # Wait, LinkItem __init__ doesn't add to scene in my previous code!
                        # Fix: explicitly add to scene.
                        link = LinkItem(self.start_city, end_city, cap, cost)
                        self.addItem(link)
                    
                    self.start_city.setSelected(False)
                    self.start_city = None
            else:
                if self.start_city:
                    self.start_city.setSelected(False)
                    self.start_city = None
                    
        super().mousePressEvent(event)

# --- Dialogs ---
class LinkDialog(QDialog):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Link Configuration")
        self.resize(350, 180)
        self.setStyleSheet("background-color: #2b2b3b; color: white;")
        layout = QFormLayout(self)
        
        self.cap_input = QSpinBox()
        self.cap_input.setRange(0, 999999)
        self.cap_input.setSuffix(" Gbps")
        self.cap_input.setValue(10)
        self.cap_input.setStyleSheet("padding: 5px; background: #3b3b4f; border: 1px solid #555;")
        
        self.cost_input = QDoubleSpinBox()
        self.cost_input.setRange(0, 999999)
        self.cost_input.setSuffix(" DT/Gbps")
        self.cost_input.setValue(5.0)
        self.cost_input.setStyleSheet("padding: 5px; background: #3b3b4f; border: 1px solid #555;")
        
        layout.addRow("Base Capacity:", self.cap_input)
        layout.addRow("Augmentation Cost:", self.cost_input)
        
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addRow(buttons)

    def get_data(self):
        return self.cap_input.value(), self.cost_input.value()

class AddDemandDialog(QDialog):
    def __init__(self, city_names):
        super().__init__()
        self.setWindowTitle("Add Traffic Demand")
        self.setStyleSheet("background-color: #2b2b3b; color: white;")
        self.resize(350, 180)
        layout = QFormLayout(self)
        
        self.src_combo = QComboBox()
        self.src_combo.addItems(city_names)
        self.src_combo.setStyleSheet("padding: 5px; background: #3b3b4f; color: white; border: 1px solid #555;")
        
        self.dst_combo = QComboBox()
        self.dst_combo.addItems(city_names)
        self.dst_combo.setStyleSheet("padding: 5px; background: #3b3b4f; color: white; border: 1px solid #555;")
        
        self.traffic_input = QSpinBox()
        self.traffic_input.setRange(1, 999999)
        self.traffic_input.setSuffix(" Gbps")
        self.traffic_input.setValue(10)
        self.traffic_input.setStyleSheet("padding: 5px; background: #3b3b4f; border: 1px solid #555;")
        
        layout.addRow("From:", self.src_combo)
        layout.addRow("To:", self.dst_combo)
        layout.addRow("Traffic:", self.traffic_input)
        
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addRow(buttons)

    def get_data(self):
        return self.src_combo.currentText(), self.dst_combo.currentText(), self.traffic_input.value()

class DataTableWindow(QDialog):
    def __init__(self, scene):
        super().__init__()
        self.setWindowTitle("Network Data Overview")
        self.resize(800, 500)
        self.setStyleSheet("background-color: #2b2b3b; color: white;")
        layout = QVBoxLayout(self)
        
        self.table = QTableWidget()
        self.table.setColumnCount(4)
        self.table.setHorizontalHeaderLabels(["Origin", "Destination", "Type", "Details"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.setStyleSheet("""
            QTableWidget { background-color: #1e1e2e; gridline-color: #333; selection-background-color: #3a86ff; }
            QHeaderView::section { background-color: #2a2a3c; padding: 5px; border: none; }
        """)
        self.table.setAlternatingRowColors(False)
        layout.addWidget(self.table)
        
        self.load_data(scene)
        
    def load_data(self, scene):
        items = [item for item in scene.items() if isinstance(item, (LinkItem, DemandItem))]
        self.table.setRowCount(len(items))
        for row, item in enumerate(items):
            self.table.setItem(row, 0, QTableWidgetItem(item.city1.name))
            self.table.setItem(row, 1, QTableWidgetItem(item.city2.name))
            
            if isinstance(item, LinkItem):
                self.table.setItem(row, 2, QTableWidgetItem("Link (Cable)"))
                cap_text = f"Cap: {item.base_capacity}G"
                if item.extra_capacity > 0:
                    cap_text += f"+{item.extra_capacity}G"
                self.table.setItem(row, 3, QTableWidgetItem(f"{cap_text} | Cost: {item.cost}"))
            elif isinstance(item, DemandItem):
                self.table.setItem(row, 2, QTableWidgetItem("Demand (Traffic)"))
                self.table.setItem(row, 3, QTableWidgetItem(f"Req: {item.traffic}G"))

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("FiberOptic Designer Pro")
        self.resize(1280, 800)
        
        # Setup Scene
        self.scene = NetworkScene(self)
        self.view = QGraphicsView(self.scene)
        self.view.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.view.setViewportUpdateMode(QGraphicsView.ViewportUpdateMode.FullViewportUpdate)
        self.view.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setCentralWidget(self.view)
        
        # Setup Toolbar
        self.create_toolbar()
        
        self.set_mode("Select")

    def create_toolbar(self):
        toolbar = QToolBar("Main Tools")
        toolbar.setIconSize(QSize(24, 24))
        toolbar.setMovable(False)
        toolbar.setStyleSheet("""
            QToolBar { background: #2a2a3c; border-bottom: 2px solid #3a86ff; spacing: 10px; padding: 5px; }
            QToolButton { color: white; background: #3b3b4f; border-radius: 4px; padding: 6px; font-weight: bold; }
            QToolButton:hover { background: #4b4b5f; }
            QToolButton:checked { background: #3a86ff; }
        """)
        self.addToolBar(Qt.ToolBarArea.TopToolBarArea, toolbar)
        
        # Action Helpers
        def btn(label, callback, checkable=False):
            action = QAction(label, self)
            action.triggered.connect(callback)
            action.setCheckable(checkable)
            toolbar.addAction(action)
            return action
        
        # Tools
        self.act_select = btn("🖐 Select", lambda: self.set_mode("Select"), True)
        self.act_city = btn("🏙 Add City", lambda: self.set_mode("AddCity"), True)
        self.act_link = btn("🔗 Add Link", lambda: self.set_mode("AddLink"), True)
        
        toolbar.addSeparator()
        btn("📉 New Demand", self.open_add_demand_dialog)
        btn("❌ Delete Selected", self.delete_selected)
        
        toolbar.addSeparator()
        btn("🚀 RUN SOLVER", self.run_solver)
        
        toolbar.addSeparator()
        btn("📋 Table", self.show_table)
        btn("💾 Save", self.save_topology)
        btn("📂 Load", self.load_topology)
        
        # Group for mode buttons
        self.mode_actions = [self.act_select, self.act_city, self.act_link]

    def set_mode(self, mode):
        self.scene.mode = mode
        
        # Visual Toggle
        for act in self.mode_actions:
            act.setChecked(False)
            
        if mode == "Select":
            self.view.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
            self.act_select.setChecked(True)
        elif mode == "AddCity":
            self.view.setDragMode(QGraphicsView.DragMode.NoDrag)
            self.act_city.setChecked(True)
        elif mode == "AddLink":
            self.view.setDragMode(QGraphicsView.DragMode.NoDrag)
            self.act_link.setChecked(True)
            
        self.statusBar().showMessage(f"Active Mode: {mode}")

    def wheelEvent(self, event):
        zoom_in = 1.15
        zoom_out = 1 / zoom_in
        factor = zoom_in if event.angleDelta().y() > 0 else zoom_out
        self.view.scale(factor, factor)

    # --- Feature Implementation ---
    
    def delete_selected(self):
        items = self.scene.selectedItems()
        if not items: return
        
        res = QMessageBox.question(self, "Confirm Delete", f"Delete {len(items)} items?", QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if res == QMessageBox.StandardButton.Yes:
            for item in items:
                self.scene.removeItem(item)
                # If city, remove its links too
                if isinstance(item, CityItem):
                    for link in item.links[:]:
                        self.scene.removeItem(link)
            self.statusBar().showMessage("Items Deleted.")

    def get_cities(self):
        return {item.name: item for item in self.scene.items() if isinstance(item, CityItem)}

    def open_add_demand_dialog(self):
        cities = self.get_cities()
        if len(cities) < 2:
            QMessageBox.warning(self, "Info", "Need 2+ cities.")
            return
        
        dialog = AddDemandDialog(sorted(cities.keys()))
        if dialog.exec():
            s, d, t = dialog.get_data()
            if s == d:
                QMessageBox.warning(self, "Error", "Source == Dest")
                return
            self.scene.addItem(DemandItem(cities[s], cities[d], t))

    def show_table(self):
        DataTableWindow(self.scene).exec()

    def find_city(self, name):
        for item in self.scene.items():
            if isinstance(item, CityItem) and item.name == name:
                return item
        # Create if missing
        import random
        c = CityItem(random.randint(200,800), random.randint(200,600), name)
        self.scene.addItem(c)
        return c

    def load_topology(self):
        fn, _ = QFileDialog.getOpenFileName(self, "Load", "", "CSV (*.csv)")
        if not fn: return
        try:
            self.scene.clear()
            self.scene.city_counter = 1
            with open(fn, 'r') as f:
                reader = csv.reader(f)
                next(reader, None)
                for row in reader:
                    if len(row) >= 4:
                        u, v, cap, cost = row[0], row[1], float(row[2]), float(row[3])
                        c1, c2 = self.find_city(u), self.find_city(v)
                        self.scene.addItem(LinkItem(c1, c2, cap, cost))
            QMessageBox.information(self, "Success", "Map Loaded.")
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def save_topology(self):
        fn, _ = QFileDialog.getSaveFileName(self, "Save Map", "", "CSV (*.csv)")
        if not fn: return
        try:
            with open(fn, 'w', newline='') as f:
                w = csv.writer(f)
                w.writerow(["Origin", "Destination", "Capacity", "Augmentation_Cost"])
                for i in self.scene.items():
                    if isinstance(i, LinkItem):
                        w.writerow([i.city1.name, i.city2.name, i.base_capacity, i.cost])
            QMessageBox.information(self, "Success", "Topology Saved.")
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def run_solver(self):
        if not solveur:
            QMessageBox.critical(self, "Error", "Solver module missing.")
            return
        
        edges, demands = [], []
        link_map = {}
        
        # Reset Capacities
        for i in self.scene.items():
            if isinstance(i, LinkItem):
                i.reset_capacity()
        
        # Gather Data
        for i in self.scene.items():
            if isinstance(i, LinkItem):
                u, v = i.city1.name, i.city2.name
                edges.append((u, v, i.base_capacity, i.cost))
                link_map[(u, v)] = i
                link_map[(v, u)] = i
            elif isinstance(i, DemandItem):
                demands.append((i.city1.name, i.city2.name, i.traffic))
        
        if not edges: return
        
        if not demands:
            QMessageBox.information(self, "Reset", "Map capacities reset.")
            return

        try:
            added, cost = solveur.solve_network(edges, demands)
            if cost == float('inf'):
                QMessageBox.warning(self, "Infeasible", "No solution.")
            else:
                msg = f"Optimal Cost: {cost:.2f}\n"
                cnt = 0
                for (u,v), val in added.items():
                    if val > 1e-6:
                        cnt += 1
                        msg += f"\n{u}->{v}: +{val}G"
                        if (u,v) in link_map:
                            link_map[(u,v)].add_extra_capacity(val)
                if cnt == 0: msg += "\nNo extra capacity needed."
                QMessageBox.information(self, "Solved", msg)
        except Exception as e:
            QMessageBox.critical(self, "Solver Error", str(e))

if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    # Global stylesheet for app-wide consistency
    app.setStyle("Fusion")
    app.setStyleSheet(f"""
        QMainWindow {{ background-color: {THEME_BG.name()}; }}
        QMessageBox {{ background-color: {THEME_BG.name()}; color: white; }}
        QLabel {{ color: white; }}
    """)
    
    w = MainWindow()
    w.show()
    sys.exit(app.exec())
