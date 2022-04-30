from django.urls import path, include
from logoDetect import views
from django.conf.urls.static import static

urlpatterns = [
    path('logo_detect/', views.LogoDetectView.as_view(), name='logodetect'),
    path('logo_url_detect/', views.LogoURLDetectView.as_view(), name='logourldetect'),
    path('train_model/', views.TrainModelView.as_view(), name='trainmodel'),
    path('update_dataset/', views.UpdateDatasetView.as_view(), name='updatedataset'),
]
