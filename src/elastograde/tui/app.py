import json,threading
from pathlib import Path
from textual.app import App,ComposeResult
from textual.widgets import Header,Footer,TabbedContent,TabPane,Static,Input,Button,RichLog
from textual import work
from elastograde.config import load
from .screens import HELP

class ElastoApp(App):
    TITLE='ElastoGrade Research'
    CSS='Screen { background: #102130; } Input { margin: 1 0; } Button { margin: 0 1; } RichLog { height: 1fr; min-height: 5; }'
    BINDINGS=[('ctrl+q','quit','Quit')]
    def compose(self)->ComposeResult:
        yield Header()
        with TabbedContent():
            with TabPane('Dashboard'):
                yield Static('Mechanics research | Clinical grading: unavailable\nNo private data is uploaded. Synthetic demo available offline.')
                yield Button('Run offline demo',id='demo')
            with TabPane('Data'):
                yield Input(placeholder='Data folder',id='data_root');yield Button('Inspect files',id='inspect')
            with TabPane('Experiment'):
                yield Input('configs/demo.yaml',id='config');yield Button('Validate / preview',id='preview');yield Button('Train',id='train')
            with TabPane('Training'):
                yield Static('Progress appears in the log. Stop requests cancellation after the current epoch. Resume uses last.pt.')
                yield Input(placeholder='Optional checkpoint for resume',id='resume');yield Button('Stop',id='stop')
            with TabPane('Inference'):
                yield Input('reports/demo/sample.npz',id='sample');yield Input(placeholder='Optional trusted checkpoint',id='checkpoint');yield Button('Infer mechanics',id='infer')
            with TabPane('Reasoning'):
                yield Input('0.8',id='prob');yield Input('10',id='cost');yield Button('Expected-loss decision',id='decide')
            with TabPane('Results'):
                yield Input('reports/demo',id='run');yield Button('Read results',id='results');yield Button('Build report',id='report')
            with TabPane('Help'):yield Static(HELP)
        yield RichLog(id='log',wrap=True,highlight=False)
        yield Footer()
    def on_mount(self):self.stop_event=threading.Event();self.busy=False
    def append_log(self,value):self.query_one('#log',RichLog).write(str(value))
    def on_button_pressed(self,event):
        key=event.button.id
        if key=='stop':self.stop_event.set();self.append_log('Cancellation requested');return
        if self.busy:self.append_log('A job is running; wait or stop training.');return
        fields={w.id:w.value for w in self.query(Input)};self.busy=True;self.stop_event.clear();self.job(key,fields)
    @work(thread=True)
    def job(self,key,f):
        try:
            if key=='demo':
                from elastograde.reporting.reports import demo
                r=demo()
            elif key=='inspect':
                from elastograde.data.inventory import inventory
                r=inventory(f['data_root'])
            elif key=='preview':r=load(f['config'])
            elif key=='train':
                from elastograde.training.engine import train
                r=train(load(f['config']),resume=f['resume'] or None,callback=lambda row:self.call_from_thread(self.append_log,row),stop_event=self.stop_event)
            elif key=='infer':
                from elastograde.data.schemas import load_sample
                from elastograde.inference.pipeline import infer
                from elastograde.inference.outputs import export
                r=export('reports/inference',infer(load_sample(f['sample']),checkpoint=f['checkpoint'] or None))
            elif key=='decide':
                from elastograde.pgm.decisions import choose,DEFAULT_LOSS
                loss=DEFAULT_LOSS.copy();loss[0,0]=float(f['cost']);r=choose(float(f['prob']),loss)
            elif key=='results':
                p=Path(f['run']);target=p/'summary.json' if (p/'summary.json').exists() else p/'status.json';r=json.loads(target.read_text())
            elif key=='report':
                from elastograde.reporting.reports import build_report
                r=build_report(f['run'])
            else:r='Unknown action'
            self.call_from_thread(self.append_log,json.dumps(r,indent=2,default=str))
        except Exception as e:self.call_from_thread(self.append_log,'FAILED: '+str(e))
        finally:self.busy=False
