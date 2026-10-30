"""Shared display text for polarity, peak current and ROI field statistics."""
from html import escape

ANODE_COLOR='#9b2635'
CATHODE_COLOR='#173b80'
FREQUENCY_NOTE='本软件的导联场模型中，电流频率不影响电场幅度；频率设定需要结合实际情况。'
TES_NOTE='tES 电流为直流值或交流峰值；交流按同频、0°/180°相位计算。幅值优化未指定神经方向或直流极性效应。'


def stimulation_note(result):
    return TES_NOTE+' '+FREQUENCY_NOTE if (result or {}).get('stimulation_type')=='tES' else FREQUENCY_NOTE


def roi_statistics_text(result):
    stats=result.get('statistics',{}).get('roi')
    if not stats:return '目标区域场强：尚未选择靶区'
    return f"目标区域均值 {stats['mean_v_m']:.4f} V/m，峰值 {stats['max_v_m']:.4f} V/m"


def electrode_scheme_html(result,names):
    lines=[]
    if result.get('stimulation_type')=='tES':
        for label,current in zip(result['electrode_labels'],result['currents_ma']):
            positive=float(current)>0; color=ANODE_COLOR if positive else CATHODE_COLOR
            name=escape(str(names.get(label,label)))
            lines.append(f'<span style="background-color:{color};color:white"> {"正向" if positive else "负向"} {name} </span> '
                         f'<b style="font-size:14px;color:#f2f5ff">{float(current):+.4f} mA（峰值）</b>')
        rows=[]
        for start in range(0,len(lines),2):
            cells=lines[start:start+2]
            rows.append('<tr>'+''.join('<td>'+cell+'</td>' for cell in cells)+'</tr>')
        total=sum(abs(float(c)) for c in result['currents_ma'])
        return '<table width="100%" cellspacing="2">'+''.join(rows)+'</table>'+f'<span style="color:#bec7d7">tES · {len(result["electrode_labels"])} 电极 · 电流绝对值合计 {total:.4f} mA</span>'
    for channel,(pair,current) in enumerate(zip(result['pairs'],result['currents_peak_ma']),1):
        positive=escape(str(names.get(pair[0],pair[0])));negative=escape(str(names.get(pair[1],pair[1])))
        lines.append(f'<span style="color:#bec7d7">通道{channel}</span> '
            f'<span style="background-color:{ANODE_COLOR};color:white"> 阳极 {positive} </span> → '
            f'<span style="background-color:{CATHODE_COLOR};color:white"> 阴极 {negative} </span> '
            f'<b style="font-size:14px;color:#f2f5ff">±{float(current):g} mA（峰值）</b>')
    return '<br>'.join(lines)
