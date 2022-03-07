from django.shortcuts import render

from django.core.paginator import Paginator, EmptyPage, PageNotAnInteger
from .utils import get_db_handle, get_collection_handle
from django.utils.functional import cached_property
from bson.json_util import dumps
from recommender.process.query_process import QueryProcess

# Create your views here.



try:
    db_handle, mongo_client = get_db_handle("arxiv", "localhost", 25000, None, None)
except:
    print("An exception occurred")

all_articles = list(db_handle["new_papers"].find())



def home(request):


    num = 12

    if request.GET.get('q') != None:
        q = request.GET.get('q') 
        qp = QueryProcess()
        paperIds = qp.run(q)

        movs = list(db_handle["new_papers"].find( { "paperId" : { "$in" : paperIds } } ))
        paginator = Paginator(movs , num)


        page_range = 0

        context =  {'articles': movs, 'page_range':page_range } 
        return render(request , 'home/home.html', context)
    else:
        q = ''


    page = request.GET.get('page', 1)

    



    paginator = Paginator(all_articles, num)

    try:
        movs = paginator.page(page)
    except PageNotAnInteger:
        movs = paginator.page(1)
    except EmptyPage:
        movs = paginator.page(paginator.num_pages)

    page_range = paginator.get_elided_page_range(number=page)

    context =  {'articles': movs, 'page_range':page_range } 
    return render(request , 'home/home.html', context)



def article(request, pk):

    article =  list(db_handle["new_papers"].find({"paperId": pk })).pop()
    context = {'id': pk , 'article':article}
    return render(request, 'home/article.html', context)

