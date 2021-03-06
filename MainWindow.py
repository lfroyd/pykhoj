
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from future_builtins import *

import os
import platform

from PyQt4 import QtCore, QtGui
from pyqtgraph.dockarea import *

from paramtreewidget import DesignParameters
import plotwidgets
from blades import Blade

__version__ = "0.0.1"


class Window(QtGui.QMainWindow):

    def __init__(self, parent=None):
        self.blade = None
        self.params = None
        super(Window, self).__init__(parent)
        self.filename = None
        ## TODO: Find out how singleshot actually works and implement it so that the GUI window launches faster
        self.setWindowTitle("Khoj 2016")
        self.setWindowState(QtCore.Qt.WindowMaximized)
        QtCore.QTimer.singleShot(0,self.create_widgets)
        self.create_actions()
        self.load_settings()


    def test(self):
        pass

    def create_widgets(self):
        self.area = DockArea()
        self.setCentralWidget(self.area)

        ## Create docks, place them into the window one at a time.
        ## Note that size arguments are only a suggestion; docks will still have to
        ## fill the entire dock area and obey the limits of their internal widgets.
        self.d1 = Dock("Design parameters", size=(150, 1), closable=False) ## give this dock the minimum possible size
        self.d2 = Dock("Triangles", size=(400,300))
        self.d3 = Dock("Axial view", size=(400,300))
        self.d4 = Dock("Radial view", size=(400,300))
        self.d5 = Dock("G-H plane", size=(400,300))
        self.d6 = Dock("3D blade", size=(400,300))
        self.d7 = Dock("3D turbine", size=(400,300))
        self.d8 = Dock("Acc. dist.", size=(400,200))
        self.d9 = Dock("Energy dist.", size=(400,200))
        self.d10 = Dock("Beta dist.", size=(400,200))
        self.d11 = Dock("Thickness", size=(400,200))
        self.d12 = Dock("Angle plot", size=(200,800))
        self.d13 = Dock("Energy plot", size=(200,800))
        self.d14 = Dock("CFD domain", size=(200,800))
        self.d15 = Dock("LE/TE plot", size=(200,800))

        ## Layout docks
        self.area.addDock(self.d1,'left')
        self.area.addDock(self.d12,'right')
        self.area.addDock(self.d2,'right',self.d1)
        self.area.addDock(self.d8,'bottom',self.d2)
        self.area.addDock(self.d3,'below',self.d2)
        self.area.addDock(self.d4,'below',self.d2)
        self.area.addDock(self.d5,'below',self.d4)
        self.area.addDock(self.d6,'below',self.d5)
        self.area.addDock(self.d7,'below',self.d6)
        self.area.addDock(self.d9,'below',self.d8)
        self.area.addDock(self.d10,'below',self.d9)
        self.area.addDock(self.d11,'below',self.d10)
        self.area.addDock(self.d13,'below',self.d12)
        self.area.addDock(self.d14,'below',self.d13)
        self.area.addDock(self.d15,'below',self.d14)

        ## Customize docks
        self.d1.hideTitleBar()
        self.d3.raiseDock()
        self.d8.raiseDock()

        # print(self.d2.isHidden(),self.d3.isHidden())utl

        ## Define widgets
        self.params = DesignParameters(self.d1)

        self.triangles = plotwidgets.TriangleWidget()
        self.axialView = plotwidgets.AxialViewWidget(self.params)
        self.accelerationControl = plotwidgets.AccelerationControlWidget(self.params)
        self.radialView = plotwidgets.RadialViewWidget(self.params)
        self.ghPlane = plotwidgets.ghPlaneWidget(self.params)
        self.betaControl = plotwidgets.BetaControlWidget(self.params)
        self.energyControl = plotwidgets.EnergyControlWidget(self.params)
        self.blade3D = plotwidgets.blade3DWidget(self.params)

        # create dict of the various plotting axes and respective docks
        self.dockWidgetDict = {'Param Tree': (self.params.tree, self.d1),
                               'Axial View': (self.axialView, self.d3),
                               'Triangles': (self.triangles, self.d2),
                               'Acceleration': (self.accelerationControl, self.d8),
                               'Beta Distribution': (self.betaControl, self.d10),
                               'Energy Distribution': (self.energyControl, self.d9),
                               'Radial View': (self.radialView, self.d4),
                               'GH Plane': (self.ghPlane, self.d5),
                               'Blade 3D': (self.blade3D, self.d6)
                               }

        ## Add widgets into each dock
        self.addDockContent(self.dockWidgetDict)

        ## Pass axis references to Params (and later Blade)
        self.params.setPlotAxes(self.dockWidgetDict)


    def addDockContent(self,inputDict):
        """ Add widgets to each dock according to input dict """
        for key in inputDict:
            inputDict[key][1].addWidget(inputDict[key][0])


    def create_actions(self):
        fileNewAction = self.createAction("&New...", self.fileNew,
                QtGui.QKeySequence.New, "filenew", "New project")
        fileOpenAction = self.createAction("&Open...", self.fileOpen,
                QtGui.QKeySequence.Open, "fileopen",
                "Open existing project")
        fileSaveAction = self.createAction("&Save", self.fileSave,
                QtGui.QKeySequence.Save, None, "Save project")
        fileSaveAsAction = self.createAction("Save &As...",
                self.fileSaveAs, None,
                tip="Save project using a new name")
        fileQuitAction = self.createAction("&Quit", self.close,
                "Ctrl+Q", None, "Close the application")
        self.fileMenu = self.menuBar().addMenu("&File")
        self.fileMenuActions = (fileNewAction, fileOpenAction,
                fileSaveAction, fileSaveAsAction, None,
                fileQuitAction)
        self.fileMenu.aboutToShow.connect(self.updateFileMenu)


    def load_settings(self):
        settings = QtCore.QSettings()
        self.recentFiles = settings.value("RecentFiles").toStringList()
        self.restoreGeometry(
                settings.value("MainWindow/Geometry").toByteArray())
        self.restoreState(settings.value("MainWindow/State").toByteArray())


    def createAction(self, text, slot=None, shortcut=None, icon=None,
                     tip=None, checkable=False):
        action = QtGui.QAction(text, self)
        if icon is not None:
            action.setIcon(QtGui.QIcon(":/{0}.png".format(icon)))
        if shortcut is not None:
            action.setShortcut(shortcut)
        if tip is not None:
            action.setToolTip(tip)
            action.setStatusTip(tip)
        if slot is not None:
            action.triggered.connect(slot)
        if checkable:
            action.setCheckable(True)
        return action


    def addActions(self, target, actions):
        for action in actions:
            if action is None:
                target.addSeparator()
            else:
                target.addAction(action)

    def closeEvent(self, event):
        if self.okToContinue():
            settings = QtCore.QSettings()
            filename = (QtCore.QVariant(QtCore.QString(self.filename))
                        if self.filename is not None else QtCore.QVariant())
            settings.setValue("LastFile", filename)
            recentFiles = (QtCore.QVariant(self.recentFiles)
                           if self.recentFiles else QtCore.QVariant())
            settings.setValue("RecentFiles", recentFiles)
            settings.setValue("MainWindow/Geometry", QtCore.QVariant(
                              self.saveGeometry()))
            settings.setValue("MainWindow/State", QtCore.QVariant(
                              self.saveState()))
        else:
            event.ignore()


    def okToContinue(self):
        return True # remove this line to re-enable file saving
        if self.tree.dirty:
            reply = QtGui.QMessageBox.question(self,
                    "Khoj - Unsaved Changes",
                    "Save unsaved changes?",
                    QtGui.QMessageBox.Yes| QtGui.QMessageBox.No| QtGui.QMessageBox.Cancel)
            if reply == QtGui.QMessageBox.Cancel:
                return False
            elif reply == QtGui.QMessageBox.Yes:
                return self.fileSave()
        return True


    def loadInitialFile(self):
        settings = QtCore.QSettings()
        fname = unicode(settings.value("LastFile").toString())
        if fname and QtCore.QFile.exists(fname):
            self.loadFile(fname)


    def updateStatus(self, message):
        self.statusBar().showMessage(message, 5000)
        self.listWidget.addItem(message)
        if self.filename is not None:
            self.setWindowTitle("Khoj - {0}[*]".format(
                                os.path.basename(self.filename)))
        elif not self.image.isNull():
            self.setWindowTitle("Khoj - Unnamed[*]")
        else:
            self.setWindowTitle("Khoj[*]")
        self.setWindowModified(self.dirty)


    def updateFileMenu(self):
        self.fileMenu.clear()
        self.addActions(self.fileMenu, self.fileMenuActions[:-1])
        current = (QtCore.QString(self.filename)
                   if self.filename is not None else None)
        recentFiles = []
        for fname in self.recentFiles:
            if fname != current and QtCore.QFile.exists(fname):
                recentFiles.append(fname)
        if recentFiles:
            self.fileMenu.addSeparator()
            for i, fname in enumerate(recentFiles):
                action = QtGui.QAction(QtGui.QIcon(),
                        "&{0} {1}".format(i + 1, QtCore.QFileInfo(
                        fname).fileName()), self)
                action.setData(QtCore.QVariant(fname))
                action.triggered.connect(self.loadFile)
                self.fileMenu.addAction(action)
        self.fileMenu.addSeparator()
        self.fileMenu.addAction(self.fileMenuActions[-1])


    def fileNew(self):
        if not self.okToContinue():
            return
        # dialog = newimagedlg.NewImageDlg(self)
        # if dialog.exec_():
        #     self.addRecentFile(self.filename)
        #     self.image = QtGui.QImage()
        #     for action, check in self.resetableActions:
        #         action.setChecked(check)
        #     self.image = dialog.image()
        #     self.filename = None
        #     self.dirty = True
        #     self.showImage()
        #     self.sizeLabel.setText("{0} x {1}".format(self.image.width(),
        #                                               self.image.height()))
        #     self.updateStatus("Created new image")


    def fileOpen(self):
        if not self.okToContinue():
            return
        self.params.load()

        # dir = (os.path.dirname(self.filename)
        #        if self.filename is not None else ".")
        # formats = (["*.{0}".format(unicode(format).lower())
        #         for format in QImageReader.supportedImageFormats()])
        # fname = unicode(QFileDialog.getOpenFileName(self,
        #         "Image Changer - Choose Image", dir,
        #         "Image files ({0})".format(" ".join(formats))))
        # if fname:
        #     self.loadFile(fname)


    # def loadFile(self, fname=None):
    #     if fname is None:
    #         action = self.sender()
    #         if isinstance(action, QtGui.QAction):
    #             fname = unicode(action.data().toString())
    #             if not self.okToContinue():
    #                 return
    #         else:
    #             return
    #     if fname:
    #         self.filename = None
    #         image = QtGui.QImage(fname)
    #         if image.isNull():
    #             message = "Failed to read {0}".format(fname)
    #         else:
    #             self.addRecentFile(fname)
    #             self.image = QtGui.QImage()
    #             for action, check in self.resetableActions:
    #                 action.setChecked(check)
    #             self.image = image
    #             self.filename = fname
    #             self.showImage()
    #             self.dirty = False
    #             self.sizeLabel.setText("{0} x {1}".format(
    #                                    image.width(), image.height()))
    #             message = "Loaded {0}".format(os.path.basename(fname))
    #         self.updateStatus(message)


    def addRecentFile(self, fname):
        if fname is None:
            return
        if not self.recentFiles.contains(fname):
            self.recentFiles.prepend(QtCore.QString(fname))
            while self.recentFiles.count() > 9:
                self.recentFiles.takeLast()


    def fileSave(self):
        self.params.save()
        # if self.image.isNull():
        #     return True
        # if self.filename is None:
        #     return self.fileSaveAs()
        # else:
        #     if self.image.save(self.filename, None):
        #         self.updateStatus("Saved as {0}".format(self.filename))
        #         self.dirty = False
        #         return True
        #     else:
        #         self.updateStatus("Failed to save {0}".format(
        #                           self.filename))
        #         return False


    def fileSaveAs(self):
        self.params.save()
        # if self.image.isNull():
        #     return True
        # fname = self.filename if self.filename is not None else "."
        # formats = (["*.{0}".format(unicode(format).lower())
        #         for format in QImageWriter.supportedImageFormats()])
        # fname = unicode(QFileDialog.getSaveFileName(self,
        #         "Image Changer - Save Image", fname,
        #         "Image files ({0})".format(" ".join(formats))))
        # if fname:
        #     if "." not in fname:
        #         fname += ".png"
        #     self.addRecentFile(fname)
        #     self.filename = fname
        #     return self.fileSave()
        # return False


    # def filePrint(self):
    #     if self.image.isNull():
    #         return
    #     if self.printer is None:
    #         self.printer = QtGui.QPrinter(QtGui.QPrinter.HighResolution)
    #         self.printer.setPageSize(QtGui.QPrinter.Letter)
    #     form = QtGui.QPrintDialog(self.printer, self)
    #     if form.exec_():
    #         painter = QtGui.QPainter(self.printer)
    #         rect = painter.viewport()
    #         size = self.image.size()
    #         size.scale(rect.size(), QtCore.Qt.KeepAspectRatio)
    #         painter.setViewport(rect.x(), rect.y(), size.width(),
    #                             size.height())
    #         painter.drawImage(0, 0, self.image)



    # def helpAbout(self):
    #     QtGui.QMessageBox.about(self, "About Khoj",
    #             """<b>Khoj</b> v {0}
    #             <p>Copyright &copy; 2012-16 NTNU.
    #             All rights reserved.
    #             <p>This application can be used for
    #             preliminary design of Francis turbines.
    #             <p>Python {1} - Qt {2} - PyQt {3} on {4}""".format(
    #             __version__, platform.python_version(),
    #             QtCore.QT_VERSION_STR, QtCore.PYQT_VERSION_STR,
    #             platform.system()))


    # def helpHelp(self):
    #     form = helpform.HelpForm("index.html", self)
    #     form.show()
