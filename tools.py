from langchain.tools import BaseTool

class ImageCaptionTool(BaseTool):
    name = None
    description = None
    
    def _run(self, img_path):
        pass
    
    def _arun(self, query:str):
        pass

class ObjectDetectionTool(BaseTool):
    name = None
    description = None
    
    def _run(self, img_path):
        pass
    
    def _arun(self, query:str):
        pass