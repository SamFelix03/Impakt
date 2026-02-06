import opik

client = opik.Opik()
print(f"✓ Connected to Opik")
print(f"Workspace: {client.workspace_name}")

try:
    traces = list(client.get_traces(project_name='disaster-monitoring', limit=10))
    print(f"\nTotal traces in 'disaster-monitoring': {len(traces)}")
    
    if traces:
        print("\nRecent traces:")
        for i, trace in enumerate(traces[:5], 1):
            print(f"{i}. Trace ID: {trace.id}")
            print(f"   Start time: {trace.start_time}")
            print(f"   Project: {trace.project_name}")
            print()
    else:
        print("\n⚠️  No traces found in 'disaster-monitoring' project")
        print("   Possible reasons:")
        print("   1. Traces are still syncing (wait 10-30 seconds and try again)")
        print("   2. Wrong project name")
        print("   3. Wrong workspace")
        
    # Check all projects
    print("\nChecking all projects in workspace:")
    projects = client.get_projects()
    for project in projects[:10]:
        print(f"  - {project.name}")
        
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
