import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "websocket-client"))

from krita import DockWidgetFactory, DockWidgetFactoryBase, Krita
from .defaults import (
    TAB_CONFIG,
    TAB_GENERATE,
    TAB_PREVIEW,
    TAB_SDCOMMON,
    TAB_UPSCALE,
    TAB_CONTROLNET,
    TAB_WORKFLOW
)
from .docker import create_docker
from .extension import SDPluginExtension
from .pages import (
    ConfigPage,
    SDCommonPage,
    UpscalePage,
    ControlNetPage,
    WorkflowPage,
    GeneratePage
)
from .pages.preview import PreviewPage
from .script import script
from .utils import reset_docker_layout

instance = Krita.instance()
instance.addExtension(SDPluginExtension(instance))
instance.addDockWidgetFactory(
    DockWidgetFactory(
        TAB_SDCOMMON,
        DockWidgetFactoryBase.DockLeft,
        create_docker(SDCommonPage),
    )
)
instance.addDockWidgetFactory(
    DockWidgetFactory(
        TAB_GENERATE,
        DockWidgetFactoryBase.DockLeft,
        create_docker(GeneratePage),
    )
)
instance.addDockWidgetFactory(
    DockWidgetFactory(
        TAB_UPSCALE,
        DockWidgetFactoryBase.DockLeft,
        create_docker(UpscalePage),
    )
)
instance.addDockWidgetFactory(
    DockWidgetFactory(
        TAB_WORKFLOW,
        DockWidgetFactoryBase.DockLeft,
        create_docker(WorkflowPage),
    )
)
instance.addDockWidgetFactory(
    DockWidgetFactory(
        TAB_CONTROLNET,
        DockWidgetFactoryBase.DockLeft,
        create_docker(ControlNetPage),
    )
)
instance.addDockWidgetFactory(
    DockWidgetFactory(
        TAB_CONFIG,
        DockWidgetFactoryBase.DockLeft,
        create_docker(ConfigPage),
    )
)
instance.addDockWidgetFactory(
    DockWidgetFactory(
        TAB_PREVIEW,
        DockWidgetFactoryBase.DockLeft,
        create_docker(PreviewPage),
    )
)


# dumb workaround to ensure its only created once
if script.cfg("first_setup", bool):
    instance.notifier().windowCreated.connect(reset_docker_layout)
    script.cfg.set("first_setup", False)
