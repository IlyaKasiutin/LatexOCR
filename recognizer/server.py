from concurrent import futures
import logging


import grpc
from grpc_reflection.v1alpha import reflection
import recognizer_pb2_grpc as recognizer_pb2_grpc
import recognizer_pb2 as recognizer_pb2


from latex_ocr import LatexOCR


class RecognizerServicer(recognizer_pb2_grpc.RecognizerServicer):
    def __init__(self):
        self.latex_ocr = LatexOCR()

    def RecognizeFormula(self, request, context):
        formula = self.latex_ocr.convert_formula(request[0])
        logging.log(logging.WARN, type(request.data))
        return recognizer_pb2.RecognizedResult(result=f'formula')

    def RecognizeMixed(self, request, context):
        mixed = self.latex_ocr.convert_mixed(request[0])
        logging.log(logging.WARN, type(request.data[0]))
        return recognizer_pb2.RecognizedResult(result=f'mixed')

    def RecognizeText(self, request, context):
        text = self.latex_ocr.convert_text(request[0])
        logging.log(logging.WARN, type(request.data[0]))
        return recognizer_pb2.RecognizedResult(result=f'text')


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    recognizer_pb2_grpc.add_RecognizerServicer_to_server(
        RecognizerServicer(), server
    )
    SERVICE_NAMES = (
        recognizer_pb2.DESCRIPTOR.services_by_name['Recognizer'].full_name,
        reflection.SERVICE_NAME,
    )
    reflection.enable_server_reflection(SERVICE_NAMES, server)
    server.add_insecure_port("[::]:50051")
    server.start()
    server.wait_for_termination()


if __name__ == "__main__":
    logging.basicConfig()
    serve()
