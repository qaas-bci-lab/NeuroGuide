"""One landscape A4 page, with fresh slices through the saved ROI center."""
import io
import os
import time
from pathlib import Path
from html import escape
import numpy as np
try:
    from reportlab.lib.pagesizes import A4,landscape
    from reportlab.lib.colors import HexColor
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.pdfgen.canvas import Canvas
    from reportlab.lib.utils import ImageReader
    from reportlab.platypus import Paragraph,Table,TableStyle,KeepInFrame
    REPORTLAB_AVAILABLE=True
except ImportError:
    REPORTLAB_AVAILABLE=False


def report_font():
    for name,path in [('NeuroGuideCJK','C:/Windows/Fonts/simhei.ttf'),('NeuroGuideYaHei','C:/Windows/Fonts/msyh.ttc')]:
        if os.path.isfile(path):
            try:
                if name not in pdfmetrics.getRegisteredFontNames():pdfmetrics.registerFont(TTFont(name,path))
                pdfmetrics.registerFontFamily(name,normal=name,bold=name,italic=name,boldItalic=name)
                return name
            except Exception:continue
    from reportlab.pdfbase.cidfonts import UnicodeCIDFont
    if 'STSong-Light' not in pdfmetrics.getRegisteredFontNames():pdfmetrics.registerFont(UnicodeCIDFont('STSong-Light'))
    return 'STSong-Light'


def report_center(engine,stimulation=None,slice_indices=None,round_to_voxel=True):
    roi=(stimulation or {}).get('roi') or {}
    if 'center_ras_mm' in roi:
        center=(np.linalg.inv(engine.affine) @ np.r_[roi['center_ras_mm'],1])[:3]
    elif roi.get('mask'):
        import nibabel as nib
        image=nib.load(roi['mask'])
        if image.shape!=engine.data.shape or not np.allclose(image.affine,engine.affine,atol=1e-4):
            raise ValueError('报告靶区与当前 MRI 网格不一致。')
        data=np.asarray(image.dataobj)
        mask=np.isin(data,roi['labels']) if roi.get('labels') else data>0
        points=np.argwhere(mask)
        if not len(points):raise ValueError('报告靶区为空。')
        center=points.mean(axis=0)
    else:
        regions=[ov['data'] for ov in engine.overlays if ov.get('is_roi')]
        points=np.argwhere(regions[0]>0) if regions else np.empty((0,3))
        center=points.mean(axis=0) if len(points) else (slice_indices if slice_indices is not None else np.asarray(engine.data.shape)//2)
    if not np.isfinite(center).all():raise ValueError('报告靶区中心无效。')
    center=np.rint(center).astype(int) if round_to_voxel else np.asarray(center,dtype=float)
    return tuple(np.clip(center,0,np.asarray(engine.data.shape)-1))


def slice_image(engine,axis,center,overlays):
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    selected=[slice(None)]*3;selected[axis]=center[axis];selected=tuple(selected)
    orient=lambda data:np.flipud(np.rot90(data))
    fig=Figure(figsize=(3.55,3.05),facecolor='white');FigureCanvasAgg(fig)
    ax=fig.add_axes([.015,.025,.80,.87]);ax.set_facecolor('black');ax.axis('off')
    from matplotlib.font_manager import FontProperties
    font_path=next((path for path in ('C:/Windows/Fonts/simhei.ttf','C:/Windows/Fonts/msyh.ttc') if os.path.isfile(path)),None)
    ax.set_title(['矢状面','冠状面','轴位面'][axis],fontsize=11,color='#182538',pad=6,
        fontproperties=FontProperties(fname=font_path) if font_path else None)
    spacing=np.linalg.norm(engine.affine[:3,:3],axis=0) if engine.affine is not None else np.ones(3)
    other=[index for index in range(3) if index!=axis];aspect=spacing[other[1]]/spacing[other[0]]
    ax.imshow(orient(engine.data[selected]),cmap='gray',origin='lower',aspect=aspect,
        vmin=engine.display_min,vmax=engine.display_max,interpolation='nearest')
    field=None
    for ov in overlays:
        data=orient(ov['data'][selected])
        if ov.get('is_roi'):
            region=np.isfinite(data)&(data>0)
            if region.any() and not region.all():ax.contour(region.astype(float),levels=[.5],colors=['#ffd38b'],linewidths=.7)
        elif not ov.get('is_atlas'):
            lower=ov.get('display_min',ov.get('min',0));upper=ov.get('display_max',ov.get('max',1))
            values=np.ma.masked_where((~np.isfinite(data))|(data<=lower),data)
            field=ax.imshow(values,cmap=ov.get('cmap','jet'),alpha=ov.get('alpha',1.),origin='lower',
                vmin=lower,vmax=upper,aspect=aspect,interpolation='bilinear')
    if getattr(engine,'show_locator',True):
        ax.axvline(center[other[0]],color='#8ab4ff',linewidth=.6,linestyle='--')
        ax.axhline(center[other[1]],color='#8ab4ff',linewidth=.6,linestyle='--')
    if field is not None:
        cax=fig.add_axes([.86,.16,.025,.57]);bar=fig.colorbar(field,cax=cax)
        bar.ax.tick_params(labelsize=7);bar.ax.set_title('V/m',fontsize=8,pad=6)
    buffer=io.BytesIO();fig.savefig(buffer,format='png',dpi=190,facecolor='white');buffer.seek(0)
    return buffer


def generate_report(engine,base_path,disease,target,output_path=None,canvases=None,patient_info=None):
    if not REPORTLAB_AVAILABLE:return False,'缺少 reportlab 库。'
    if engine.data is None:return False,'请先加载 MRI 和仿真电场。'
    info=patient_info or {};stimulation=info.get('stimulation') or {}
    fields=[ov for ov in engine.overlays if not ov.get('is_atlas') and not ov.get('is_roi')]
    if not fields:return False,'请先加载或计算仿真电场，再导出报告。'
    if output_path is None:
        safe_name=''.join('_' if char in '<>:"/\\|?*' else char for char in info.get('name','')).strip()
        output_path=str(Path.home()/'Desktop'/f"NeuroGuide_{safe_name or 'report'}_{time.strftime('%Y%m%d_%H%M%S')}.pdf")
    try:
        from ui.result_text import stimulation_note,ANODE_COLOR,CATHODE_COLOR
        from ui.individual import electrode_names,model_root
        font=report_font();width,height=landscape(A4);margin=38;usable=width-2*margin
        pdf=Canvas(str(output_path),pagesize=(width,height));pdf.setTitle('NeuroGuide 仿真报告')
        normal=ParagraphStyle('body',fontName=font,fontSize=10,leading=14,textColor=HexColor('#253349'))
        def paragraph(text,x,y,w,h,size=10):
            style=ParagraphStyle('block',parent=normal,fontSize=size,leading=size*1.4)
            block=KeepInFrame(w,h,[Paragraph(text,style)],mode='shrink')
            block.wrapOn(pdf,w,h);block.drawOn(pdf,x,y+h-block.height)
        def heading(text,y):
            pdf.setFont(font,12);pdf.setFillColor(HexColor('#203e6b'));pdf.drawString(margin,y,text)
        pdf.setFont(font,21);pdf.setFillColor(HexColor('#182c49'));pdf.drawString(margin,height-35,'NeuroGuide 仿真报告')
        pdf.setStrokeColor(HexColor('#326bd1'));pdf.line(margin,height-46,width-margin,height-46)
        heading('患者基本信息',height-67)
        pdf.setFillColor(HexColor('#f0f3f8'));pdf.roundRect(margin,height-123,usable,44,4,fill=1,stroke=0)
        for index,(label,value) in enumerate([('姓名',info.get('name')),('性别',info.get('gender')),('年龄',info.get('age'))]):
            paragraph(escape(label+'：'+(value or '未填写')),margin+9+index*usable/3,height-99,usable/3-18,18)
        paragraph(escape('患病类型：'+(info.get('disease_type') or disease or '未填写')),margin+9,height-121,usable-18,18)
        heading('仿真结果方案',height-147)
        names=electrode_names(model_root());scheme=[]
        tes=stimulation.get('stimulation_type')=='tES'
        for channel,(pair,current) in enumerate(zip(stimulation.get('pairs',[]),stimulation.get('currents_peak_ma',[])),1):
            scheme.append([f'通道 {channel}',names.get(pair[0],str(pair[0])),names.get(pair[1],str(pair[1])),f'{float(current):g} mA（峰值）'])
        tes_values=[]
        if tes:
            tes_values=list(zip(stimulation['electrode_labels'],stimulation['currents_ma']))
            for start in range(0,len(tes_values),2):
                row=[]
                for label,current in tes_values[start:start+2]:row.extend([names.get(label,str(label)),f'{float(current):+.5f} mA（峰值）'])
                scheme.append(row+['']*(4-len(row)))
        if not scheme:scheme=[['方案',target or '未记录','未记录',info.get('current') or '未记录']]
        rows=[['tES 电极','电流（峰值）','tES 电极','电流（峰值）'] if tes else ['通道','阳极','阴极','电流峰值'],*scheme]
        row_height=min(20,64/len(scheme)) if tes else 24
        cell_style=ParagraphStyle('tes_cells',parent=normal,fontSize=min(10,row_height*.65),leading=min(14,row_height*.85)) if tes else normal
        data=[]
        for row_index,row in enumerate(rows):
            cells=[]
            for column,value in enumerate(row):
                color=ANODE_COLOR if row_index and column==1 else CATHODE_COLOR if row_index and column==2 else '#253349'
                if tes:
                    index=(row_index-1)*2+column//2
                    color=(ANODE_COLOR if tes_values[index][1]>0 else CATHODE_COLOR) if row_index and index<len(tes_values) else '#253349'
                cells.append(Paragraph(f'<font color="{color}">'+escape(str(value))+'</font>',cell_style))
            data.append(cells)
        view=Table(data,colWidths=[usable*.25]*4 if tes else [usable*.12,usable*.30,usable*.30,usable*.28],rowHeights=[20]+[row_height]*len(scheme))
        view.setStyle(TableStyle([('VALIGN',(0,0),(-1,-1),'MIDDLE'),('BACKGROUND',(0,0),(-1,0),HexColor('#e2eaf6')),
            ('BACKGROUND',(0,1),(-1,-1),HexColor('#f6f8fb')),('LEFTPADDING',(0,0),(-1,-1),9),
            ('LINEBELOW',(0,0),(-1,-1),.4,HexColor('#d6dfec'))]))
        view.wrapOn(pdf,usable,height);view.drawOn(pdf,margin,height-160-20-row_height*len(scheme))
        center=report_center(engine,stimulation,info.get('slice_indices'),round_to_voxel=False)
        xyz=engine.affine @ np.r_[center,1] if engine.affine is not None else np.r_[center,1]
        roi=stimulation.get('roi') or {};roi_name='、'.join(roi.get('region_names',[])) or target or '当前靶区'
        heading('ROI 场强（峰值幅度）',height-251)
        paragraph(escape(roi_name)+f' · 中心 ({xyz[0]:.1f}, {xyz[1]:.1f}, {xyz[2]:.1f}) mm',margin,height-274,usable,19)
        stats=stimulation.get('statistics',{}).get('roi')
        text=f"均值 <b>{stats['mean_v_m']:.4f} V/m</b>　　峰值 <b>{stats['max_v_m']:.4f} V/m</b>" if stats else 'ROI 场强未记录'
        paragraph(text,margin,height-294,usable,19,size=11)
        from core.display_geometry import report_display_engine
        render_engine=report_display_engine(engine)
        if getattr(engine,'view',None) is not None:
            center=tuple(np.clip(np.rint(engine.view.native_voxel_to_view(center)).astype(int),0,np.asarray(render_engine.data.shape)-1))
        else:
            center=tuple(np.clip(np.rint(center).astype(int),0,np.asarray(render_engine.data.shape)-1))
        fields=[ov for ov in render_engine.overlays if not ov.get('is_atlas') and not ov.get('is_roi')]
        overlays=list(fields[:1] if stimulation else fields)
        regions=[ov for ov in render_engine.overlays if ov.get('is_roi')]
        if getattr(engine,'clip_fields_to_roi',False) and regions:
            region=regions[-1]['data']>0
            overlays=[{**ov,'data':np.where(region,ov['data'],np.nan)} for ov in overlays]
        if getattr(engine,'show_roi',True):overlays+=regions
        column=(usable-16)/3
        for axis in range(3):
            image=slice_image(render_engine,axis,center,overlays)
            pdf.drawImage(ImageReader(image),margin+axis*(column+8),61,column,230,preserveAspectRatio=True,anchor='c',mask='auto')
        paragraph(escape(stimulation_note(stimulation)),margin,29,usable,22,size=9)
        pdf.setFont(font,7);pdf.setFillColor(HexColor('#8a95a5'));pdf.drawRightString(width-margin,14,time.strftime('%Y-%m-%d %H:%M'))
        pdf.showPage();pdf.save()
        return True,str(output_path)
    except Exception as exc:
        return False,f'生成报告失败：{exc}'
