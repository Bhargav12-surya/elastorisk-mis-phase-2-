from typer.testing import CliRunner
from elastograde.cli import app
runner=CliRunner()
def test_help():assert runner.invoke(app,['--help']).exit_code==0
def test_cli_bad_input():assert runner.invoke(app,['infer','--input','absent.npz']).exit_code!=0
def test_cli_pgm():assert runner.invoke(app,['pgm','demo','--topic','moralization']).exit_code==0
def test_cli_demo(tmp_path):
 r=runner.invoke(app,['demo','--out',str(tmp_path)]);assert r.exit_code==0,r.output;assert (tmp_path/'report.html').exists()
def test_tui_mount():
 import asyncio
 from elastograde.tui.app import ElastoApp
 async def run():
  app=ElastoApp()
  async with app.run_test(size=(100,40)) as pilot:
   await pilot.pause();assert app.query_one('#sample').value=='reports/demo/sample.npz'
 asyncio.run(run())
