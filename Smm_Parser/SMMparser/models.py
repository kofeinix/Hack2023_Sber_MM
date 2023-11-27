from django.db import models
from uuid import uuid4
# Create your models here.
from django.utils.safestring import mark_safe
from django.urls import reverse
# from django.contrib.sites.models import Site
# domain = Site.objects.get_current().domain


class Pages(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid4, editable=False)
    categ_root = models.CharField(max_length=100, null=True, blank=True)
    categ1 = models.CharField(max_length=100, null=True, blank=True)
    categ2 = models.CharField(max_length=100, null=True, blank=True)
    categ3 = models.CharField(max_length=100, null=True, blank=True)
    categ4 = models.CharField(max_length=100, null=True, blank=True)
    name = models.CharField(max_length=100, unique=True)
    url = models.CharField(max_length=100, null=True, blank=True)
    properties = models.TextField(null=True, blank=True)

    def __str__(self):
        return self.name

    def link2(self):
        view_name = f"admin:{self._meta.app_label}_pages_change"
        link_url = reverse(view_name, args=[self.id])
        # return format_html('<a href="{}">{}</a>', link_url, obj.page)
        return link_url


class Images(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid4, editable=False)
    url = models.CharField(max_length=500, unique=True)
    page = models.ForeignKey(to=Pages, on_delete=models.CASCADE)
    imageio = models.BinaryField(blank=True, null=True)

    def get_html_photo(self):
        # return mark_safe(f"<img src='{self.url}' width=150/>")
        style = "<style type='text/css'> .custom-class { background-color:green; } .custom-class:hover { background-color:Red; } </style>"
        t = "custom-class:hover img {  background-color: yellow; opacity: 0.5;}"
        style = f"<style type='text/css'> {t}</style>"
        return mark_safe(
            f"{style} <a class='custom-class' href='{self.url}'> <img src='{self.url}' width=150/> </a>"
        )

    def get_url_href(self):
        # style = "<style type='text/css'> .custom-class { background-color:green; } .custom-class:hover { background-color:Red; } </style>"
        return mark_safe(f"<a class='custom-class' href='{self.url}'>{self.url}</a>")

    # def io_image(self, obj):
    #     return mark_safe("<img src='/get_image' />")