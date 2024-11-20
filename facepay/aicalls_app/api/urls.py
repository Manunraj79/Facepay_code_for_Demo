from django.urls import path,include
from aicalls_app.api import views
from rest_framework.routers import DefaultRouter

# router = DefaultRouter()
# router.register(r'registration', views.UserRegistrationViewSet, basename='register')


urlpatterns = [
    path('train/', views.TrainView.as_view()),
    path('test/', views.TestView.as_view()),
]