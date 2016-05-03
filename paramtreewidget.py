import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtGui
from pyqtgraph.parametertree import Parameter, ParameterTree
from pyqtgraph.parametertree import types as pTypes
import pyqtgraph.configfile
import sys, os
from math import pi, sqrt, atan2
from triangles import Triangle
from PyQt4 import QtCore

from blades import Blade


class DesignParameters(QtGui.QWidget):
    def __init__(self, parent=None):
        super(DesignParameters, self).__init__(parent)
        self.dirty = False
        self.tree = ParameterTree(showHeader=False)
        self.objectGroup = ObjectGroupParam()
        # self.avAxis = axes['Axial View']
        # self.blade = blade
        self.blade = Blade()
        self.needsUpdate = True
        self.g = 9.81
        self.stateLoaded = False
        self.updateFinished = False

        self.params = Parameter.create(name='params', type='group', children=[
            dict(name='Turbine Type', type='list', values=['Francis']),
            #dict(name='Unit System', type='list', values=['', 'MKS']),
            #dict(name='Duration', type='float', value=10.0, step=0.1, limits=[0.1, None]),
            #dict(name='Reference Frame', type='list', values=[]),
            #dict(name='Update Continuous', type='bool', value=False),
            #dict(name='Animation Speed', type='float', value=1.0, dec=True, step=0.1, limits=[0.0001, None]),
            # dict(name='Recalculate', type='action'),
            # dict(name='Save', type='action'),
            # dict(name='Load', type='action'),
            self.objectGroup,
            ])

        self.tree.setParameters(self.params, showTop=False)
        # self.params.param('Recalculate').sigActivated.connect(self.recalculate)
        # self.params.param('Save').sigActivated.connect(self.save)
        # self.params.param('Load').sigActivated.connect(self.load)
        # self.params.param('Load Preset..').sigValueChanged.connect(self.loadPreset)
        self.params.sigTreeStateChanged.connect(self.treeChanged)

        ## read list of preset configs
        # presetDir = os.path.join(os.path.abspath(os.path.dirname(sys.argv[0])), 'presets')
        # if os.path.exists(presetDir):
        #     presets = [os.path.splitext(p)[0] for p in os.listdir(presetDir)]
        #     self.params.param('Load Preset..').setLimits(['']+presets)



    def treeChanged(self, *args):
        #print("tree changes:")
        self.dirty = True
        change = None
        for param, change, data in args[1]:
            path = self.params.childPath(param)
            if path is not None:
                # childName = '.'.join(path)
                childName = param.name()
            else:
                childName = param.name()
            print('  parameter: %s'% childName)
            print('  change:    %s'% change)
            print('  data:      %s'% str(data))
            print('  ----------')

            if change=='value': #and self.updateFinished:
                self.recalculate()
                # elif change == 'childRemoved':
                # Do something?


    def save(self):
        fn = str(pg.QtGui.QFileDialog.getSaveFileName(self, "Save turbine..", "untitled.cfg", "Config Files (*.cfg)"))
        if fn == '':
            return
        state = self.params.saveState()
        pg.configfile.writeConfigFile(state, fn)


    def load(self):
        fn = str(pg.QtGui.QFileDialog.getOpenFileName(self, "Load turbine..", "", "Config Files (*.cfg)"))
        if fn == '':
            return
        self.stateLoaded = False
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
        self.params.param('Design Parameters').clearChildren()
        self.params.restoreState(state, removeChildren=True)
        self.reconnect()
        self.updateFinished = True
        self.needsUpdate = True

        ## Initialize or reset Blade control curves
        if self.blade.accCurve is None:
            self.blade.initAcceleration()
            self.blade.initBetaDistribution()
        else:
            self.blade.resetAcceleration()
            self.blade.resetBetaDistribution()

        self.recalculate()
        self.dirty = False


    def reconnect(self):
        self.gridFreq = self.params.param('Design Parameters').param('Machine Data').param('Grid frequency')
        self.rotorSpeed = self.params.param('Design Parameters').param('Machine Data').param('Rotor speed')
        self.polePairs = self.params.param('Design Parameters').param('Machine Data').param('Pole pairs')
        self.acceleration = self.params.param('Design Parameters').param('Meridional View').param('Acceleration')
        self.outletDiameter = self.params.param('Design Parameters').param('Meridional View').param('Outlet diameter D2')
        self.inletDiameter = self.params.param('Design Parameters').param('Meridional View').param('Inlet diameter D1')
        self.inletHeight = self.params.param('Design Parameters').param('Meridional View').param('Inlet height B1')
        self.shroudHeight = self.params.param('Design Parameters').param('Meridional View').param('Shroud height b')
        self.flow = self.params.param('Design Parameters').param('Machine Data').param('Flow')
        self.head = self.params.param('Design Parameters').param('Machine Data').param('Head')
        self.etah = self.params.param('Design Parameters').param('Machine Data').param('Hydraulic efficiency')

        self.U1_red = self.params.param('Design Parameters').param('Reduced Velocities').param('U1_red')
        self.U2_red = self.params.param('Design Parameters').param('Reduced Velocities').param('U2_red')
        self.CM1_red = self.params.param('Design Parameters').param('Reduced Velocities').param('CM1_red')
        self.CM2_red = self.params.param('Design Parameters').param('Reduced Velocities').param('CM2_red')
        self.W1_red = self.params.param('Design Parameters').param('Reduced Velocities').param('W1_red')
        self.W2_red = self.params.param('Design Parameters').param('Reduced Velocities').param('W2_red')
        self.C1_red = self.params.param('Design Parameters').param('Reduced Velocities').param('C1_red')
        self.alpha1 = self.params.param('Design Parameters').param('Reduced Velocities').param('Alpha1')
        self.beta1 = self.params.param('Design Parameters').param('Reduced Velocities').param('Beta1')
        self.beta2 = self.params.param('Design Parameters').param('Reduced Velocities').param('Beta2')

        self.params.red = sqrt(2*self.g*self.head.value())

        # self.nBladeSets = self.params.param('Design Parameters').param('Machine Data').param('Number of blade sets')
        # self.nStreamlines = self.params.param('Design Parameters').param('Meridional View').param('Number of streamlines')

        self.gridFreq.sigValueChanged.connect(self.rotorSpeedChanges)
        self.polePairs.sigValueChanged.connect(self.rotorSpeedChanges)
        self.acceleration.sigValueChanged.connect(self.accelerationChanged)
        self.outletDiameter.sigValueChanged.connect(self.recalcAcceleration)
        self.inletDiameter.sigValueChanged.connect(self.inletDiameterChanged)
        self.U1_red.sigValueChanged.connect(self.U1_redChanged)
        self.head.sigValueChanged.connect(self.headChanged)
        self.flow.sigValueChanged.connect(self.recalcAcceleration)
        self.inletHeight.sigValueChanged.connect(self.recalcAcceleration)


    def calcReducedVelocities(self):
        self.params.blockSignals(True)
        self.tree.setUpdatesEnabled(False)

        red = sqrt(2*self.g*self.head.value())
        self.CM1_red.setValue((self.flow.value()/(pi*self.inletDiameter.value()*self.inletHeight.value()))/red)
        self.CM2_red.setValue(self.CM1_red.value()*self.acceleration.value())
        CU1_red = self.etah.value()/(2*self.U1_red.value())
        self.C1_red.setValue(sqrt(CU1_red**2 + self.CM1_red.value()**2))
        self.alpha1.setValue(atan2(self.CM1_red.value(),CU1_red)*180/pi)
        WU1_red = self.U1_red.value()-CU1_red
        self.W1_red.setValue(sqrt(WU1_red**2 + self.CM1_red.value()**2))
        self.beta1.setValue(atan2(self.CM1_red.value(),WU1_red)*180/pi)
        self.U2_red.setValue(pi*self.rotorSpeed.value()*self.outletDiameter.value()/60/red)
        self.W2_red.setValue(sqrt(self.U2_red.value()**2 + self.CM2_red.value()**2))
        self.beta2.setValue(atan2(self.CM2_red.value(),self.U2_red.value())*180/pi)

        self.tree.setUpdatesEnabled(True)
        self.params.blockSignals(False)


    def dockVisibilityChanged(self):
        print('dock changed')


    def setPlotAxes(self,dockWidgetDict):
        self.triangleAxis = dockWidgetDict['Triangles'][0].axis
        self.blade.setPlotAxes(dockWidgetDict)
        self.initializeTriangles()

    def initializeTriangles(self):
        """ Initialize velocity triangle figure """
        self.inletTriangle = Triangle()
        self.triangleAxis.addItem(self.inletTriangle)
        self.outletTriangle = Triangle()
        self.triangleAxis.addItem(self.outletTriangle)
        ## Define the set of connections in each graph
        self.adj_inlet = np.array([[0,1], [1,2], [2,3], [1,3], [0,3]])
        self.adj_outlet = np.array([[0,1], [1,2], [0,2]])

        ## Define the symbol to use for each node (this is optional)
        self.symbols_inlet = ['o','o','o','o']
        self.symbols_outlet = ['o','o','o']

        ## Define the line style for each connection (this is optional)
        # self.lines_inlet = np.array([
        #     (0,0,0,255,2),
        #     (0,0,0,255,2),
        #     (0,0,0,255,2),
        #     (0,0,0,255,2),
        #     (0,0,0,255,2),
        #     ], dtype=[('red',np.ubyte),('green',np.ubyte),('blue',np.ubyte),('alpha',np.ubyte),('width',float)])

        ## Define text to show next to each symbol
        self.texts_inlet = ["Point %d" % i for i in range(4)]
        self.texts_outlet = ["Point %d" % i for i in range(3)]

        ## Define which points are draggable in x and y direction, respectively
        self.drags_inlet = np.array([[1,0],[1,0], [1,0], [0,1] ])
        self.drags_outlet = np.array([[0,0],[1,0], [0,1] ])

        ## Define which points are dependent on other points dragging {master: [dependents]}
        self.deps_inlet = {1: np.array([3])}
        self.deps_outlet = {}

        self.inletTriangle.changed.connect(self.inletTriangleChanged)
        self.outletTriangle.changed.connect(self.outletTriangleChanged)

    def inletTriangleChanged(self,isChanged):
        print('inlet tringle changed')


    def outletTriangleChanged(self, isChanged):
        print('outlet tringle changed')

    def updateTriangles(self):
        ## Define positions of nodes

        CU1_red = self.etah.value()/(2*self.U1_red.value())
        pos_inlet = np.array([
            [0,0],
            [CU1_red,0],
            [self.U1_red.value(),0],
            [CU1_red,-self.CM1_red.value()]
            ], dtype=float)

        dy = -self.CM1_red.value()*2
        pos_outlet = np.array([
            [0,0+dy],
            [self.U2_red.value(),0+dy],
            [0,-self.CM2_red.value()+dy]
            ], dtype=float)

        ## Update the graph
        self.inletTriangle.setData(pos=pos_inlet, adj=self.adj_inlet, size=0.01, symbol='o', pxMode=False, drag = self.drags_inlet, dep = self.deps_inlet) #pen=None, text=self.texts_inlet
        self.outletTriangle.setData(pos=pos_outlet, adj=self.adj_outlet, size=0.01, symbol='o', pxMode=False, drag = self.drags_outlet, dep = self.deps_outlet) #pen=None, text=self.texts_outlet


    def rotorSpeedChanges(self):
        self.rotorSpeed.setValue(self.gridFreq.value()*60/float(self.polePairs.value()))
        self.U1_redChanged()


    def inletDiameterChanged(self):
        D1 = self.inletDiameter.value()
        U1 = D1*pi*self.rotorSpeed.value()/60.
        H = self.head.value()
        red = sqrt(2*self.g*H)
        self.U1_red.setValue(U1 / red, blockSignal=self.U1_redChanged)
        self.recalcAcceleration()


    def U1_redChanged(self):
        H = self.head.value()
        red = sqrt(2*self.g*H)
        U1 = red * self.U1_red.value()
        self.inletDiameter.setValue(U1 * 60 / (self.rotorSpeed.value() * pi), blockSignal=self.inletDiameterChanged)
        self.recalcAcceleration()


    def headChanged(self):
        H = self.head.value()
        red = sqrt(2*self.g*H)
        U1 = red * self.U1_red.value()
        self.inletDiameter.setValue(U1 * 60 / (self.rotorSpeed.value() * pi), blockSignal=self.inletDiameterChanged)
        self.recalcAcceleration()


    def accelerationChanged(self):
        Q = self.flow.value()
        D1 = self.inletDiameter.value()
        B1 = self.inletHeight.value()
        Cm1 = Q/(pi*D1*B1)
        Cm2 = Cm1*self.acceleration.value()
        self.outletDiameter.setValue(sqrt(4 * Q / (pi * Cm2)), blockSignal=self.recalcAcceleration)


    def recalcAcceleration(self):
        Q = self.flow.value()
        D1 = self.inletDiameter.value()
        B1 = self.inletHeight.value()
        Cm1 = Q/(pi*D1*B1)
        Cm2 = Q/(pi * self.outletDiameter.value() ** 2 / 4.)
        self.acceleration.setValue(Cm2/Cm1, blockSignal=self.accelerationChanged)


    def recalculate(self):
        print('recalculate')
        if self.blade is not None:
            if 1: #self.needsUpdate:
                self.calcReducedVelocities()
                self.blade.readParams(self.params)
                self.blade.rebuildBlade()
                self.updateTriangles()
                self.needsUpdate = False
        else:
            pass


class ObjectGroupParam(pTypes.GroupParameter):
    def __init__(self):
        pTypes.GroupParameter.__init__(self, name="Design Parameters", addText="Add New..", addList=['Splitter'])

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


pTypes.registerParameterType('Splitter', SplitterParam)
