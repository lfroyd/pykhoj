# import matplotlib
# matplotlib.use('Qt4Agg')
import functools
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d import Axes3D

from PyQt4.QtGui import QMenu, QAction
import pyqtgraph as pg


class TriangleWidget(pg.GraphicsLayoutWidget):

    def __init__(self, parent=None):
        pg.GraphicsLayoutWidget.__init__(self,parent)
        self.setAspectLocked(True)
        self.setBackground('w')
        self.setAntialiasing(True)
        self.axis = self.addViewBox()
        self.axis.setAspectLocked()

class AccelerationControlWidget(FigureCanvas):

    def __init__(self, params, parent=None):
        self.figure = Figure(facecolor=(1., 1., 1.))
        super(AccelerationControlWidget, self).__init__(self.figure)
        self.setParent(parent)
        self.axes = self.figure.add_subplot(111)#, aspect='equal')
        self.axes.set_autoscale_on(True)
        self.canvas = self.figure.canvas
        self.blade = params.blade
        self.menu = None

    def contextMenuEvent(self, event):
        if self.menu is None:
            self.menu = QMenu(self)
            self.reset = QAction("Reset", self.menu)
            self.reset.triggered[()].connect(self.blade.resetAcceleration)
            self.menu.addAction(self.reset)
            self.reset.setCheckable(False)
        self.action = self.menu.exec_(self.mapToGlobal(event.pos()))



class AxialViewWidget(FigureCanvas):

    def __init__(self, params, parent=None):
        self.figure = Figure(facecolor=(1., 1., 1.))
        super(AxialViewWidget, self).__init__(self.figure)
        self.setParent(parent)
        self.axes = self.figure.add_subplot(111, aspect='equal')
        self.axes.set_autoscale_on(True)
        self.canvas = self.figure.canvas
        self.blade = params.blade
        self.menu = None

    def contextMenuEvent(self, event):
        if self.menu is None:
            self.menu = QMenu(self)

            self.setLE =  QAction("Set LE curve", self.menu)
            self.setTE =  QAction("Set TE curve", self.menu)

            self.setTE.toggled.connect(self.blade.addTECurve)
            self.setLE.toggled.connect(self.blade.addLECurve)

            self.menu.addAction(self.setTE)
            self.menu.addAction(self.setLE)

            self.setTE.setCheckable(True)
            self.setLE.setCheckable(True)

        self.action = self.menu.exec_(self.mapToGlobal(event.pos()))


class RadialViewWidget(FigureCanvas):

    def __init__(self, params, parent=None):
        self.figure = Figure(facecolor=(1., 1., 1.))
        super(RadialViewWidget, self).__init__(self.figure)
        self.setParent(parent)
        self.axes = self.figure.add_subplot(111, projection='polar')
        self.axes.set_autoscale_on(False)
        self.axes.set_rlabel_position(315)
        self.canvas = self.figure.canvas
        self.blade = params.blade
        self.menu = None


class ghPlaneWidget(FigureCanvas):

    def __init__(self, params, parent=None):
        self.figure = Figure(facecolor=(1., 1., 1.))
        super(ghPlaneWidget, self).__init__(self.figure)
        self.setParent(parent)
        self.axes = self.figure.add_subplot(111)#, aspect='equal')
        self.axes.set_autoscale_on(False)
        self.canvas = self.figure.canvas
        self.blade = params.blade
        self.menu = None


class EnergyControlWidget(FigureCanvas):
    def __init__(self, params, parent=None):
        self.figure = Figure(facecolor=(1., 1., 1.))
        super(EnergyControlWidget, self).__init__(self.figure)
        self.setParent(parent)
        self.axes = self.figure.add_subplot(111)  # , aspect='equal')
        self.axes.set_autoscale_on(False)
        self.canvas = self.figure.canvas
        self.blade = params.blade
        self.menu = None


class BetaControlWidget(FigureCanvas):
    def __init__(self, params, parent=None):
        self.figure = Figure(facecolor=(1., 1., 1.))
        super(BetaControlWidget, self).__init__(self.figure)
        self.setParent(parent)
        self.axes = self.figure.add_subplot(111)  # , aspect='equal')
        self.axes.set_autoscale_on(False)
        self.canvas = self.figure.canvas
        self.blade = params.blade
        self.menu = None

    def contextMenuEvent(self, event):
        if self.menu is None:
            self.menu = QMenu(self)
            self.reset = QAction("Reset", self.menu)
            self.reset.triggered[()].connect(self.blade.resetBetaDistribution)
            self.menu.addAction(self.reset)
            self.reset.setCheckable(False)
        self.action = self.menu.exec_(self.mapToGlobal(event.pos()))

class blade3DWidget(FigureCanvas):
    def __init__(self, params, parent=None):
        self.figure = Figure(facecolor=(1., 1., 1.), figsize=plt.figaspect(1.0))
        super(blade3DWidget, self).__init__(self.figure)
        self.setParent(parent)
        self.axes = self.figure.add_subplot(111, projection='3d')
        self.axes.set_autoscale_on(True)
        self.canvas = self.figure.canvas
        self.blade = params.blade
        self.menu = None

    def contextMenuEvent(self, event):
        if self.menu is None:
            self.menu = QMenu(self)
            self.projectionSubmenu = QMenu('Projection',self)
            self.menu.addMenu(self.projectionSubmenu)

            self.sideProjection = QAction("Side view", self.menu)
            self.sideProjection.triggered[()].connect(functools.partial(self.blade.setBlade3DProjection,'Side'))
            self.projectionSubmenu.addAction(self.sideProjection)
            self.sideProjection.setCheckable(False)

            self.topProjection = QAction("Top view", self.menu)
            self.topProjection.triggered[()].connect(functools.partial(self.blade.setBlade3DProjection,'Top'))
            self.projectionSubmenu.addAction(self.topProjection)
            self.topProjection.setCheckable(False)

            self.frontProjection = QAction("Front view", self.menu)
            self.frontProjection.triggered[()].connect(functools.partial(self.blade.setBlade3DProjection,'Front'))
            self.projectionSubmenu.addAction(self.frontProjection)
            self.frontProjection.setCheckable(False)

        self.action = self.menu.exec_(self.mapToGlobal(event.pos()))

class cascade3DWidget(FigureCanvas):
    def __init__(self, params, parent=None):
        self.figure = Figure(facecolor=(1., 1., 1.), figsize=plt.figaspect(1.0))
        super(cascade3DWidget, self).__init__(self.figure)
        self.setParent(parent)
        self.axes = self.figure.add_subplot(111, projection='3d')
        self.axes.set_autoscale_on(True)
        self.canvas = self.figure.canvas
        self.menu = None

    def contextMenuEvent(self, event):
        if self.menu is None:
            self.menu = QMenu(self)
            self.projectionSubmenu = QMenu('Projection', self)
            self.menu.addMenu(self.projectionSubmenu)
            self.cascadeSubmenu = QMenu('Cascade', self)
            self.menu.addMenu(self.cascadeSubmenu)
            self.cascadeSubmenu = QMenu('Hub', self)
            self.menu.addMenu(self.cascadeSubmenu)
            self.cascadeSubmenu = QMenu('Shroud', self)
            self.menu.addMenu(self.cascadeSubmenu)

            self.sideProjection = QAction("Side view", self.menu)
            self.sideProjection.triggered[()].connect(functools.partial(self.blade.setCascade3DProjection, 'Side'))
            self.projectionSubmenu.addAction(self.sideProjection)
            self.sideProjection.setCheckable(False)

            self.topProjection = QAction("Top view", self.menu)
            self.topProjection.triggered[()].connect(functools.partial(self.blade.setCascade3DProjection, 'Top'))
            self.projectionSubmenu.addAction(self.topProjection)
            self.topProjection.setCheckable(False)

            self.frontProjection = QAction("Front view", self.menu)
            self.frontProjection.triggered[()].connect(functools.partial(self.blade.setCascade3DProjection, 'Front'))
            self.projectionSubmenu.addAction(self.frontProjection)
            self.frontProjection.setCheckable(False)

        self.action = self.menu.exec_(self.mapToGlobal(event.pos()))