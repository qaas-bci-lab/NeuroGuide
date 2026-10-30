"""Live per-stage elapsed times, separate from the technical execution log."""
import time
from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import QLabel

NAMES={'tissue_segmentation':'组织分割','electrode_placement':'电极放置',
       'leadfield_prediction':'导联场预测','mri_preparation':'影像准备','atlas_registration':'图谱配准',
       'field_synthesis':'电场合成','electrode_optimization':'靶区优化'}


def duration_text(seconds):
    seconds=max(0,int(seconds))
    if seconds < 60: return f'{seconds} 秒'
    minutes,seconds=divmod(seconds,60)
    if minutes < 60: return f'{minutes} 分 {seconds:02d} 秒'
    hours,minutes=divmod(minutes,60)
    return f'{hours} 小时 {minutes:02d} 分'


class StageTimes(QLabel):
    def __init__(self,parent=None):
        super().__init__(parent)
        self.setObjectName('stageTimes'); self.setWordWrap(True)
        self.setStyleSheet('color: #a8afbd; font-size: 11px; padding: 2px 0;')
        self.records={}; self.timer=QTimer(self); self.timer.setInterval(1000)
        self.timer.timeout.connect(self.render); self.render()

    def reset(self):
        self.records={}; self.timer.stop(); self.render()

    def update_stage(self,event):
        stage=event['stage']
        if stage not in NAMES: return
        record=dict(self.records.get(stage,{}))
        record.update(event)
        if event['state']=='running': record['started']=time.monotonic()
        self.records[stage]=record
        if any(r.get('state')=='running' for r in self.records.values()): self.timer.start()
        else: self.timer.stop()
        self.render()

    def restore(self,records):
        self.timer.stop()
        self.records={k:dict(v) for k,v in records.items() if k in NAMES}
        self.render()

    def stopped(self):
        for record in self.records.values():
            if record.get('state')=='running':
                record.update(seconds=time.monotonic()-record['started'],state='stopped')
        self.timer.stop(); self.render()

    def render(self):
        parts=[]
        for stage,name in NAMES.items():
            if stage not in self.records and stage not in ('tissue_segmentation','electrode_placement','leadfield_prediction'): continue
            record=self.records.get(stage,{})
            state=record.get('state'); seconds=record.get('seconds')
            if state=='running':
                text='运行中 '+duration_text(time.monotonic()-record['started'])
            elif state=='cached':
                text='已复用'+('（原耗时 '+duration_text(seconds)+'）' if seconds is not None else '')
            elif seconds is not None:
                text=duration_text(seconds)+('（未完成）' if state in ('failed','stopped') else '')
            else: text='—'
            device=record.get('device','')
            if device: text+=' / '+('GPU' if str(device).startswith('cuda') else 'CPU')
            parts.append(name+'：'+text)
        self.setText('   ·   '.join(parts))
