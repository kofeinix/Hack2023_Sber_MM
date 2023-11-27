from django.contrib import admin
from django.utils.html import format_html
from django.urls import reverse
from django.utils.safestring import mark_safe
from .models import Pages, Images


@admin.register(Pages)
class PagesAdmin(admin.ModelAdmin):
    list_display = ["name", 'self_name', 'categ_root',
                    # "categ1", "categ2", "categ3", "categ4",
                    "photos"
                    ]
    search_fields = ['name']
    fields = ['categ_root', "categ1", "categ2", "categ3", "categ4", 'self_name', 'properties',
              # "urls_",
              "photos"]
    readonly_fields = ["self_name", "categ_root", "categ1", "categ2", "categ3", "categ4", "properties",  "photos", ]
    list_per_page = 10
    list_filter = ['categ_root', "categ1", "categ2", "categ3", "categ4"]
    save_on_top = True
    list_display_links = ("name",)

    def self_name(self, obj):
        """ по имени можно сразу перейти к товару """
        return mark_safe(f"<a href='{obj.url}'>{obj.name}</a>")

    self_name.short_description = "name"

    # def urls_(self, obj):
    #     return ", ".join([
    #         i.url for i in obj.images_set.all()
    #     ])
    # urls_.short_description = "URLS"

    def photos(self, obj):
        if not obj.images_set.all():
            return "-"

        result = "".join(
            [i.get_html_photo() for i in obj.images_set.all()]
                    )
        return mark_safe(result)

    photos.short_description = "photos"
    photos.allow_tags = True


@admin.register(Images)
class ImagesAdmin(admin.ModelAdmin):
    list_display = ['page', "link_", 'get_url_href', "get_html_photo",
                    # "io_image"
                    ]
    list_display_links = ("page",)
    readonly_fields = ['page', "link_", 'get_url_href', "get_html_photo",
                       "imageio"
                       ]
    ordering = ['page']
    search_fields = ['page__name']

    def link_(self, obj):
        view_name = f"admin:{obj._meta.app_label}_pages_change"
        link_url = reverse(view_name, args=[obj.page_id])
        return format_html('<a href="{}">{}</a>', link_url, obj.page)

        # работает, но урл длинный
        # url = f"{obj._meta.app_label}/pages/{obj.page.id}/change/"
        # return format_html(f'<a href="{url}"> {obj.page} </>')
        # http://localhost:8000/admin/SMMparser/pages/7878bacb-4fc4-4d27-983e-f2aae829e078/change/
    link_.short_description = 'url_to_admin'

    # def get_html_photo(self, obj):
    #     return mark_safe(f"<img src='{obj.url}' width=150>")
    # get_html_photo.short_description = "миниатюра"  # для того чтобы переопределить название в таблице


