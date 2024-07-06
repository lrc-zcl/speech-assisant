from pyecharts import options as opts
from pyecharts.charts import Geo
from pyecharts.globals import ChartType, SymbolType

c = (
    Geo()
        .add_schema(maptype="china")
        .add(
        "",
        [("信阳", 1), ("南京", 20230902), ("徐州", 20231201), ("无锡", 20240501), ("上海", 20240503), ("嘉兴", 0),
          ("长沙", 0), ("重庆", 0), ("西双版纳", 0), ("昆明", 0), ("大理", 0),("丽江", 0),
         ("成都", 0), ("西安", 0), ("威海", 0), ("青岛", 0), ("北京", 0),("杭州", 0)],
        type_=ChartType.EFFECT_SCATTER,
        color="white",
    )
        .add(
        "geo",
        [("信阳", "南京"), ("南京", "徐州"), ("徐州", "无锡"), ("无锡", "上海")],
        type_=ChartType.LINES,
        effect_opts=opts.EffectOpts(
            symbol=SymbolType.ARROW, symbol_size=6, color="blue"
        ),
        linestyle_opts=opts.LineStyleOpts(curve=0.2, color="blue"),
    )
        .add(
        "geo",
        [("上海", "嘉兴"), ("嘉兴","杭州" ),("杭州", "长沙"),("杭州", "长沙"),("长沙", "西双版纳"), ("西双版纳", "昆明"), ("昆明", "大理"),( "大理","丽江"),( "丽江","重庆"),("重庆", "成都"),( "成都","西安"),( "西安","青岛"),("青岛", "威海"),("威海","北京")],
        type_=ChartType.LINES,
        effect_opts=opts.EffectOpts(
            symbol=SymbolType.ARROW, symbol_size=6, color="pink"
        ),
        linestyle_opts=opts.LineStyleOpts(curve=0.2, color="pink"),
    )
        .set_series_opts(label_opts=opts.LabelOpts(is_show=False))
        .set_global_opts(title_opts=opts.TitleOpts(title="Geo-Lines"))
        .render("geo_lines.html")
)
