import os
import threading as td


def run_simulation(geckoport: int = 43036):
    print(geckoport)
    os.environ['JDK_HOME'] = '/usr/lib/jvm/java-19-openjdk/'
    os.environ['CLASSPATH'] = '/home/spock/.apps/gecko/GeckoCIRCUITS.jar'
    print('import jnius')
    from jnius import autoclass
    
    JString = autoclass('java.lang.String')
    Inst = autoclass('gecko.GeckoRemoteObject')

    simfilepath = '../circuits/DAB_MOSFET_Modulation_Lm_nlC.ipes'
    # geckoport = 43036
    # Note: absolute filepaths needed. Otherwise, there will occur java.lang.String error when using relative paths
    simfilepath = os.path.abspath(simfilepath)
    # Start GeckoCIRCUITS. This opens the Gecko window:
    mydata = td.local()
    print('Inst.startNewRemoteInstance(geckoport)')
    mydata.ginst = Inst.startNewRemoteInstance(geckoport)
    # Open the simulation file. Use java-strings:
    fname = JString(simfilepath)
    print('ginst.openFile(fname)')
    mydata.ginst.openFile(fname)
    
    # No pre-simulation:
    # ginst.set_dt_pre(0)
    # ginst.set_Tend_pre(0)
    # ginst.set_dt(timestep)  # Simulation time step
    # ginst.set_Tend(simtime)  # Simulation time
    print('Run gecko simulation')
    mydata.ginst.runSimulation()

    print('Shutting down gecko')
    mydata.ginst.shutdown()


def gecko_thread():
    # mutex = td.Lock()
    threads = []
    # Start the worker threads
    for i in range(3):
        geckoport = 43010 + i
        # t = td.Thread(target=run_simulation(), kwargs={'geckoport': geckoport}, name=str(i))
        kwargs = {'geckoport': geckoport}
        t = td.Thread(target=run_simulation, kwargs=kwargs, name=str(i))
        t.start()
        threads.append(t)
        print(i, geckoport)
        print(t)

    # Wait for the threads to complete
    for t in threads:
        t.join()

# ---------- MAIN ----------
if __name__ == '__main__':
    print("Start of Gecko thread test ...")

    # os.environ['JDK_HOME'] = '/usr/lib/jvm/java-19-openjdk/'
    # os.environ['CLASSPATH'] = '/home/spock/.apps/gecko/GeckoCIRCUITS.jar'
    # print('import jnius')
    # from jnius import autoclass

    gecko_thread()