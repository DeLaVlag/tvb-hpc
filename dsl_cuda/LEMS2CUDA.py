# from models import G2DO
from mako.template import Template

import os
import sys
import NeuroML
sys.path.append("{}".format(os.path.dirname(NeuroML.__file__)))

from lems.model.model import Model

# model file location
# modelname = 'Oscillator'
# modelname = 'Kuramoto'
# modelname = 'rWongWang'
modelname = 'Epileptor'

fp_xml = '../NeuroML/' + modelname.lower() + '_CUDA' + '.xml'

model = Model()
model.import_from_file(fp_xml)
# modelextended = model.resolve()

def templating():

    # drift dynamics
    # modelist = list()
    # modelist.append(model.component_types[modelname])

    modellist = model.component_types[modelname]

    # coupling functionality
    couplinglist = list()
    # couplinglist.append(model.component_types['coupling_function_pop1'])

    for i, cplists in enumerate(model.component_types):
        if 'coupling' in cplists.name:
            couplinglist.append(cplists)

    # collect all signal amplification factors per state variable.
    # signalampl = list()
    # for i, sig in enumerate(modellist.dynamics.derived_variables):
    #     if 'sig' in sig.name:
    #         signalampl.append(sig)

    # collect total number of exposures combinations.
    expolist = list()
    for i, expo in enumerate(modellist.exposures):
        for chc in expo.choices:
            expolist.append(chc)

    # print((couplinglist[0].dynamics.derived_variables['pre'].expression))
    #
    # for m in range(len(couplinglist)):
    #     # print((m))
    #     for k in (couplinglist[m].functions):
    #         print(k)

    # only check whether noise is there, if so then activate it
    noisepresent=False
    for ct in (model.component_types):
        if ct.name == 'noise' and ct.description == 'on':
            noisepresent=True

    # start templating
    template = Template(filename='tmpl8_CUDA.py')
    model_str = template.render(
                            modelname=modelname,
                            const=modellist.constants,
                            dynamics=modellist.dynamics,
                            params=modellist.parameters,
                            coupling=couplinglist,
                            noisepresent=noisepresent,
                            expolist=expolist
                            )
    # write template to file
    modelfile="../models/" + modelname.lower() + ".c"
    with open(modelfile, "w") as f:
        f.writelines(model_str)


templating()