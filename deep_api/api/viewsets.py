from rest_framework import viewsets
from rest_framework.views import APIView
from rest_framework.filters import SearchFilter
from api.serializers import deepModelSerializer
from rest_framework.response import Response
from rest_framework import status
from api.models import deepModel
from django.conf import settings
from api.utils import prepare_image
from keras.applications import imagenet_utils
from PIL import Image
import io

class deepModelViewSet(viewsets.ModelViewSet):
    queryset = deepModel.objects.all()
    filter_backends = [SearchFilter]
    search_fields = ['=name']
    serializer_class = deepModelSerializer
from keras.models import load_model

class PredictAPIView (APIView):
    def get (self, request):
        model = settings.NN_MODELS.get('MLP')
        print(model.summary())
        return Response({'success':model.summary()})

    def post(self, request):
        #El uso del modelo.
        model = settings.NN_MODELS.get('MLP')
        image_request = request.FILES['number'].read()
        image_request = Image.open(io.BytesIO(image_request))
        image = prepare_image(image_request, target=(28, 28))
        
        try:
            with settings.GRAPH.as_default():
                predictions = model.predict(image)
        except Exception as error:
            return Response({'message': str(error)}, 
                status=status.HTTP_400_BAD_REQUEST)
        return Response({'success': predictions})
