import pyqtgraph as pg
from pyqtgraph.Qt import QtGui, QtCore
from pyqtgraph.parametertree import Parameter, ParameterTree
from pyqtgraph.parametertree import types as pTypes
import pyqtgraph.configfile
from PyQt4.QtCore import (QTimer)
import numpy as np
import user
import collections
import sys, os

import matplotlib
matplotlib.use('Qt4Agg')

from matplotlib.lines import Line2D
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from beziercurves import BezierBuilder


class Mpwidget(FigureCanvas):
    def __init__(self, parent=None):

        self.figure = Figure(facecolor=(1., 1., 1.))
        super(Mpwidget, self).__init__(self.figure)
        self.setParent(parent)
        self.axes = self.figure.add_subplot(111)

        # Empty line
        line = Line2D([], [], ls='--', c='#666666',
                      marker='o', mew=2, mec='#204a87')
        # Initial control points
        xvals=[0.1 ,0.2, 0.3, 0.4, 0.5]
        yvals=[0.1,0.9,-0.3,0.5,-0.1]

        line.set_data(xvals,yvals)
        self.axes.add_line(line)

        # Canvas limits
        self.axes.set_xlim(-1, 1)
        self.axes.set_ylim(-1, 1)
        self.axes.set_title("Bezier curve")

        # Create BezierBuilder
        self.bezier_builder = BezierBuilder(line,self.axes)


class KhojGUI(QtGui.QWidget):
    def __init__(self):
        QtGui.QWidget.__init__(self)
        self.setupGUI()
        self.objectGroup = ObjectGroupParam()

        self.params = Parameter.create(name='params', type='group', children=[
            #dict(name='Load Preset..', type='list', values=[]),
            #dict(name='Unit System', type='list', values=['', 'MKS']),
            #dict(name='Duration', type='float', value=10.0, step=0.1, limits=[0.1, None]),
            #dict(name='Reference Frame', type='list', values=[]),
            #dict(name='Update Continuous', type='bool', value=False),
            #dict(name='Animation Speed', type='float', value=1.0, dec=True, step=0.1, limits=[0.0001, None]),
            dict(name='Recalculate', type='action'),
            dict(name='Save', type='action'),
            dict(name='Load', type='action'),
            self.objectGroup,
            ])
        self.tree.setParameters(self.params, showTop=False)
        self.params.param('Recalculate').sigActivated.connect(self.recalculate)
        self.params.param('Save').sigActivated.connect(self.save)
        self.params.param('Load').sigActivated.connect(self.load)
        #self.params.param('Load Preset..').sigValueChanged.connect(self.loadPreset)
        self.params.sigTreeStateChanged.connect(self.treeChanged)

        ## read list of preset configs
        presetDir = os.path.join(os.path.abspath(os.path.dirname(sys.argv[0])), 'presets')
        if os.path.exists(presetDir):
            presets = [os.path.splitext(p)[0] for p in os.listdir(presetDir)]
            self.params.param('Load Preset..').setLimits(['']+presets)

    def setupGUI(self):
        self.layout = QtGui.QVBoxLayout()
        self.layout.setContentsMargins(0,0,0,0)
        self.setLayout(self.layout)
        self.splitter = QtGui.QSplitter()
        self.splitter.setOrientation(QtCore.Qt.Horizontal)
        self.layout.addWidget(self.splitter)

        self.tree = ParameterTree(showHeader=False)
        self.splitter.addWidget(self.tree)

        #self.splitter2 = QtGui.QSplitter()
        self.TurbinePlot = Mpwidget()

        #self.splitter2.setOrientation(QtCore.Qt.Horizontal)
        self.splitter.addWidget(self.TurbinePlot)

        #self.TurbinePlot = pg.GraphicsLayoutWidget()
        #self.splitter2.addWidget(self.worldlinePlots)

        #self.animationPlots = pg.GraphicsLayoutWidget()
        #self.splitter.addWidget(self.TurbinePlot)
        #self.splitter2.addWidget(self.TurbinePlot)

        #self.splitter.setSizes([int(self.width()*0.3), int(self.width()*0.7)])

        #self.PlotLeft = self.TurbinePlot.addPlot()
        #self.PlotRight = self.TurbinePlot.addPlot()
        #
        # self.inertAnimationPlot = self.animationPlots.addPlot()
        # self.inertAnimationPlot.setAspectLocked(1)
        # self.refAnimationPlot = self.animationPlots.addPlot()
        # self.refAnimationPlot.setAspectLocked(1)
        #
        # self.inertAnimationPlot.setXLink(self.inertWorldlinePlot)
        # self.refAnimationPlot.setXLink(self.refWorldlinePlot)

    def treeChanged(self, *args):
        print('Tree changed')
        # clocks = []
        # for c in self.params.param('Objects'):
        #     clocks.extend(c.clockNames())
        for param, change, data in args[1]:
            print param, change, data
            #if change == 'childAdded':
        #self.params.param('Reference Frame').setLimits(clocks)
        #self.setAnimation(self.params['Animate'])

    def save(self):
        fn = str(pg.QtGui.QFileDialog.getSaveFileName(self, "Save State..", "untitled.cfg", "Config Files (*.cfg)"))
        if fn == '':
            return
        state = self.params.saveState()
        pg.configfile.writeConfigFile(state, fn)

    def load(self):
        fn = str(pg.QtGui.QFileDialog.getOpenFileName(self, "Save State..", "", "Config Files (*.cfg)"))
        if fn == '':
            return
        state = pg.configfile.readConfigFile(fn)
        self.loadState(state)

    def loadPreset(self, param, preset):
        if preset == '':
            return
        path = os.path.abspath(os.path.dirname(__file__))
        fn = os.path.join(path, 'presets', preset+".cfg")
        state = pg.configfile.readConfigFile(fn)
        self.loadState(state)

    def loadState(self, state):
        if 'Load Preset..' in state['children']:
            del state['children']['Load Preset..']['limits']
            del state['children']['Load Preset..']['value']
        self.params.param('Objects').clearChildren()
        self.params.restoreState(state, removeChildren=True)
        self.recalculate()

    def recalculate(self):
        print('recalculate')


class ObjectGroupParam(pTypes.GroupParameter):
    def __init__(self):
        pTypes.GroupParameter.__init__(self, name="Objects", addText="Add New..", addList=['Splitter'])

    def addNew(self, typ):
        if typ == 'Splitter':
            self.addChild(SplitterParam())

class SplitterParam(pTypes.GroupParameter):
    def __init__(self, **kwds):
        defs = dict(name="Splitter", autoIncrementName=True, renamable=False, removable=True, children=[
            dict(name='Splitter type', type='list', value='Linked', limits=['Linked']),
            dict(name='Pitch', type='float', value=0.5, step=0.1, limits=[0.1, 0.9]),
            dict(name='Length', type='float', value=2/3., step=0.1, limits=[0.0, 1.0]),
            ])
        #defs.update(kwds)
        pTypes.GroupParameter.__init__(self, **defs)
        self.restoreState(kwds, removeChildren=False)



# pTypes.registerParameterType('Splitter', SplitterParam)

# class AccelerationGroup(pTypes.GroupParameter):
#     def __init__(self, **kwds):
#         defs = dict(name="Acceleration", addText="Add Command..")
#         pTypes.GroupParameter.__init__(self, **defs)
#         self.restoreState(kwds, removeChildren=False)
#
#     def addNew(self):
#         nextTime = 0.0
#         if self.hasChildren():
#             nextTime = self.children()[-1]['Proper Time'] + 1
#         self.addChild(Parameter.create(name='Command', autoIncrementName=True, type=None, renamable=True, removable=True, children=[
#             dict(name='Proper Time', type='float', value=nextTime),
#             dict(name='Acceleration', type='float', value=0.0, step=0.1),
#             ]))
#
#     def generate(self):
#         prog = []
#         for cmd in self:
#             prog.append((cmd['Proper Time'], cmd['Acceleration']))
#         return prog
#
# pTypes.registerParameterType('AccelerationGroup', AccelerationGroup)


if __name__ == '__main__':
    pg.mkQApp()
    #import pyqtgraph.console
    #cw = pyqtgraph.console.ConsoleWidget()
    #cw.show()
    #cw.catchNextException()
    win = KhojGUI()
    win.setWindowTitle("PyKhoj")
    win.show()
    win.resize(1100,700)

    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()


    #win.params.param('Objects').restoreState(state, removeChildren=False)

