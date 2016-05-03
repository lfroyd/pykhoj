import functools
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d import Axes3D
from PyQt4.QtGui import QMenu, QAction
import pyqtgraph as pg


class TriangleWidget(pg.GraphicsLayoutWidget):
    """ CLASS - Defines the graphics area for plotting the velocity triangles"""
    def __init__(self, parent=None):
        pg.GraphicsLayoutWidget.__init__(self,parent)
        self.setAspectLocked(True)
        self.setBackground('w')
        self.setAntialiasing(True)
        self.axis = self.addViewBox()
        self.axis.setAspectLocked()

class AccelerationControlWidget(FigureCanvas):
    """CLASS - Defines the graphics area for plotting the acceleration control bezier curve"""
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
        """ Defines the context menu (right-click)"""
        if self.menu is None:
            self.menu = QMenu(self)
            self.reset = QAction("Reset", self.menu)
            self.reset.triggered[()].connect(self.blade.resetAcceleration)
            self.menu.addAction(self.reset)
            self.reset.setCheckable(False)
        self.action = self.menu.exec_(self.mapToGlobal(event.pos()))



class AxialViewWidget(FigureCanvas):
    """CLASS - Defines the graphics area for plotting the meridional view and LE/TE contorl bezier curves """
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
        """ Defines the context menu (right-click)"""
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
    """CLASS - Defines the graphics area for plotting the radial view """
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
    """CLASS - Defines the graphics area for plotting the G-H plane view """
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
    """CLASS - Defines the graphics area for plotting the U-Cu (Energy) distribution control bezier curve """
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
    """CLASS - Defines the graphics area for plotting the Beta (Angle) distribution control bezier curve """
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
        """ Defines the context menu (right-click)"""
        if self.menu is None:
            self.menu = QMenu(self)
            self.reset = QAction("Reset", self.menu)
            self.reset.triggered[()].connect(self.blade.resetBetaDistribution)
            self.menu.addAction(self.reset)
            self.reset.setCheckable(False)
        self.action = self.menu.exec_(self.mapToGlobal(event.pos()))



class blade3DWidget(FigureCanvas):
    """CLASS - Defines the graphics area for plotting the Blade 3D projection """
    def __init__(self, params, parent=None):
        self.figure = Figure(facecolor=(1., 1., 1.), figsize=plt.figaspect(1.0))
        super(blade3DWidget, self).__init__(self.figure)
        self.setParent(parent)
        self.axes = self.figure.add_subplot(111, projection='3d')
        self.axes.set_autoscale_on(True)
        self.canvas = self.figure.canvas
        self.blade = params.blade
        self.blade.blade3D['showAxis'] = False

        self.menu = None
        self.projectionDict = {'Top': (90,0),
                               'Side': (0,0),
                               'Front': (0,90)}
        self.colorDict = {'Default': None,
                          'Z-value': 'Zmat',
                          'Radius': 'Rmat',
                          'CM velocity': 'CMmat',
                          'Theta angle': 'Tmat',
                          'Beta angle': 'Bmat'}


    def toggleAxis(self, isChecked):
        if isChecked:
            self.axes.set_axis_on()
            self.blade.blade3D['showAxis'] = True
        else:
            self.axes.set_axis_off()
            self.blade.blade3D['showAxis'] = False
        # self.canvas.flush_events()
        # self.canvas.update()
        self.canvas.draw()


    def setView(self,projection):
        self.axes.view_init(elev=self.projectionDict[projection][0], azim=self.projectionDict[projection][1])
        self.canvas.draw()


    def setColouring(self,colouring):
        self.blade.blade3D['Colour'] = self.colorDict[colouring]
        self.blade.plotBlade3D()


    def contextMenuEvent(self, event):
        """ Defines the context menu (right-click)"""
        if self.menu is None:
            self.menu = QMenu(self)

            self.showAxis = QAction('Show axis', self.menu)
            self.showAxis.toggled.connect(self.toggleAxis)
            self.showAxis.setCheckable(True)
            self.showAxis.setChecked(self.blade.blade3D['showAxis'])
            self.menu.addAction(self.showAxis)

            self.projectionSubmenu = QMenu('Projection',self)
            self.menu.addMenu(self.projectionSubmenu)

            self.sideProjection = QAction("Side view", self.menu)
            self.sideProjection.triggered[()].connect(functools.partial(self.setView,'Side'))
            self.projectionSubmenu.addAction(self.sideProjection)
            self.sideProjection.setCheckable(False)

            self.topProjection = QAction("Top view", self.menu)
            self.topProjection.triggered[()].connect(functools.partial(self.setView,'Top'))
            self.projectionSubmenu.addAction(self.topProjection)
            self.topProjection.setCheckable(False)

            self.frontProjection = QAction("Front view", self.menu)
            self.frontProjection.triggered[()].connect(functools.partial(self.setView,'Front'))
            self.projectionSubmenu.addAction(self.frontProjection)
            self.frontProjection.setCheckable(False)

            self.colouringSubmenu = QMenu('Colouring', self)
            self.menu.addMenu(self.colouringSubmenu)

            self.defaultColouring = QAction("Default", self.menu)
            self.defaultColouring.triggered[()].connect(functools.partial(self.setColouring, 'Default'))
            self.colouringSubmenu.addAction(self.defaultColouring)
            self.defaultColouring.setCheckable(False)

            self.zColouring = QAction("Z-value", self.menu)
            self.zColouring.triggered[()].connect(functools.partial(self.setColouring, 'Z-value'))
            self.colouringSubmenu.addAction(self.zColouring)
            self.zColouring.setCheckable(False)

            self.rColouring = QAction("Radius", self.menu)
            self.rColouring.triggered[()].connect(functools.partial(self.setColouring, 'Radius'))
            self.colouringSubmenu.addAction(self.rColouring)
            self.rColouring.setCheckable(False)

            self.tColouring = QAction("Theta angle", self.menu)
            self.tColouring.triggered[()].connect(functools.partial(self.setColouring, 'Theta angle'))
            self.colouringSubmenu.addAction(self.tColouring)
            self.tColouring.setCheckable(False)

            self.bColouring = QAction("Beta angle", self.menu)
            self.bColouring.triggered[()].connect(functools.partial(self.setColouring, 'Beta angle'))
            self.colouringSubmenu.addAction(self.bColouring)
            self.bColouring.setCheckable(False)

            self.CmColouring = QAction("CM velocity", self.menu)
            self.CmColouring.triggered[()].connect(functools.partial(self.setColouring, 'CM velocity'))
            self.colouringSubmenu.addAction(self.CmColouring)
            self.CmColouring.setCheckable(False)



        self.action = self.menu.exec_(self.mapToGlobal(event.pos()))



class cascade3DWidget(FigureCanvas):
    """CLASS - Defines the graphics area for plotting the Blade Cascade (Turbine) projection """
    def __init__(self, params, parent=None):
        self.figure = Figure(facecolor=(1., 1., 1.), figsize=plt.figaspect(1.0))
        super(cascade3DWidget, self).__init__(self.figure)
        self.setParent(parent)
        self.axes = self.figure.add_subplot(111, projection='3d')
        self.axes.set_autoscale_on(True)
        self.canvas = self.figure.canvas
        self.menu = None


    def contextMenuEvent(self, event):
        """ Defines the context menu (right-click)"""
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