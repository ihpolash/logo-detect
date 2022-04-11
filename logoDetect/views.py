from django.shortcuts import render

# Create your views here.
from rest_framework import permissions, serializers, status
from rest_framework.generics import GenericAPIView

# Create your views here.
from rest_framework.response import Response
import client
from rest_framework.throttling import UserRateThrottle, AnonRateThrottle


class LogoSerializer(serializers.Serializer):
    image = serializers.ImageField()

class LogoUrlSerializer(serializers.Serializer):
    url = serializers.CharField()


class TrainModelSerializer(serializers.Serializer):
    # image = Base64ImageField(max_length=None, use_url=True,)
    how_many_training_steps = serializers.IntegerField(default=1000)
    testing_percentage = serializers.IntegerField(default=10)
    learning_rate = serializers.FloatField(default=0.1)
    delete_checkpoint = serializers.BooleanField()



class LogoDetectView(GenericAPIView):
    permission_classes = [permissions.AllowAny]
    serializer_class = LogoSerializer

    def post(self, request, *args, **kwargs):
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        image = serializer.validated_data.get('image')
        response = client.get_logo_image_clssify(image)
        return Response(response, status=status.HTTP_200_OK)

class LogoURLDetectView(GenericAPIView):
    permission_classes = [permissions.AllowAny]
    serializer_class = LogoUrlSerializer

    def post(self, request, *args, **kwargs):
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        url = serializer.validated_data.get('url')
        response = client.get_logo_url_clssify(url)
        return Response(response, status=status.HTTP_200_OK)

class TrainModelView(GenericAPIView):
    permission_classes = [permissions.AllowAny]
    serializer_class = TrainModelSerializer

    def post(self, request, *args, **kwargs):
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        how_many_training_steps = serializer.validated_data.get('how_many_training_steps')
        testing_percentage = serializer.validated_data.get('testing_percentage')
        learning_rate = serializer.validated_data.get('learning_rate')
        delete_checkpoint = serializer.validated_data.get('delete_checkpoint')
        response = client.train_model(how_many_training_steps, testing_percentage, learning_rate, delete_checkpoint)
        return Response(response, status=status.HTTP_200_OK)