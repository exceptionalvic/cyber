from django.db import models
from django.utils import timezone
# from django.conf import settings
from datetime import datetime, timedelta

# Create your models here.
class PidginDB(models.Model):
	pidgin = models.CharField(max_length=500, blank=True)
	english = models.CharField(max_length=500, blank=True)
	created = models.DateTimeField(auto_now_add=True)


	class Meta:
		ordering = ['-created']

	def __str__(self):
		return self.pidgin

	def user_folder(self, request):
		return self.pidgin


class SentiDB(models.Model):
	tweet = models.CharField(max_length=500, blank=True)
	username = models.CharField(max_length=500, blank=True)
	sentiment = models.CharField(max_length=500, blank=True)
	created = models.DateTimeField(auto_now_add=True)

	class Meta:
		ordering = ['-created']

	def __str__(self):
		return self.tweet


class QueryDB(models.Model):
	keyword = models.CharField(max_length=500, blank=True)
	created = models.DateTimeField(auto_now_add=True)

	class Meta:
		ordering = ['-created']

	def __str__(self):
		return self.keyword
