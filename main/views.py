from django.shortcuts import render, redirect, get_object_or_404
from . models import PidginDB, SentiDB, QueryDB
from django.core.management import call_command
import collections
from datetime import datetime, timedelta
from django.core.paginator import Paginator
from django.db.models.functions import Now


def index(request):
	context = {}
	return render(request, 'index.html', context)


def result(request):
	if request.method == 'POST':
		get_q = request.POST.get('q')
		new_query = QueryDB(keyword=get_q)
		new_query.save()
		# print(get_q)
		call_command('analyze')
		getlist = SentiDB.objects.filter(sentiment='negative')
		sentilist = SentiDB.objects.all().exclude(tweet='nan')
		sentilist1 = SentiDB.objects.all()
		neut = sentilist1.filter(sentiment='neutral',created__gte=Now() - timedelta(hours=0.05)).order_by('-id')[:50].count()
		neg = sentilist1.filter(sentiment='negative',created__gte=Now() - timedelta(hours=0.05)).order_by('-id')[:50].count()
		pos = sentilist1.filter(sentiment='positive',created__gte=Now() - timedelta(hours=0.05)).order_by('-id')[:50].count()
		get_users = []
		for i in getlist:
			get_users.append(i.username)
		frequency = collections.Counter(get_users)
		freq1 = dict(frequency)
		# for j in freq1.values():
		# 	print(j)
		context = {
			'list': sentilist,
			'neut': neut,
			'pos': pos,
			'neg': neg,
			'freq': freq1
		}
		paginator = Paginator(sentilist, 50)
		page = request.GET.get('page')
		sentilist = paginator.get_page(page)
		return render(request,'results.html', context)
	else:
		return redirect('index')


def bullies(request):
	getlist = SentiDB.objects.filter(sentiment='negative')
	sentilist = SentiDB.objects.all().exclude(tweet='nan')
	collate_users = []
	for i in getlist:
		collate_users.append(i.username)
	get_users = collate_users
	# print(get_users)
	frequency = collections.Counter(get_users)
	freq1 = dict(frequency)
	freq2 = []	
	for key, value in freq1.items():
		if value > 4:
			freq2.append(key)
			# print(freq2)
	final_res = set(freq2)
	# print(final_res)
	# for j in freq1.values():
	# 	print(j)
	context = {
		'list': sentilist,
		'freq': final_res
	}
	paginator = Paginator(sentilist, 50)
	page = request.GET.get('page')
	sentilist = paginator.get_page(page)
	return render(request,'bullies.html', context)