"""M21e: assemble several teaching models from the same electric+gravity engine.

Only physical components, initial conditions and source notes differ by case.
No case supplies a trajectory formula, force callback or special integrator.
"""
from __future__ import annotations
import argparse
from hashlib import sha256
import json
from pathlib import Path

from conductor_sim import (FixedPointField, PointCharge, PrescribedField,
    TestChargeMotionSpec, UniformElectricField, UniformRingAxisField)
from conductor_sim.conservative_forces import UniformGravity
from conductor_sim.scenarios import PrescribedScenario, revise_scenario, sample_scenario_viewer
from conductor_sim.scenario_comparison import render_scenario_comparison, prepare_scenario_comparison
from conductor_sim.viewer import render_html

CASES = ('jee2012-proton', 'oblique-launch', 'electric-support', 'gravity-only',
         'point-plus-gravity', 'ring-plus-gravity', 'mass-revision')
MP, E_CHARGE, G = 1.67e-27, 1.6e-19, 10.
PDF_SHA = '9dff3c02a9f1848d2c3d99cb1a6e7a44dc7f5a51f7a3a2b8cc92d10ce1fbca5d'


def build_case(case):
    if case == 'mass-revision':
        case = 'oblique-launch'
    origin, velocity = (0.,0.,0.), (0.,0.,0.)
    labels = (('field:applied', 'Applied electric field'),)
    gravity = UniformGravity((0.,-G,0.))
    anchor = dict(kind='teacher-hypothetical', case=case)
    if case == 'jee2012-proton':
        source = Path(__file__).resolve().parents[1] / '02_conductors_half_size.pdf'
        if sha256(source.read_bytes()).hexdigest() != PDF_SHA:
            raise ValueError('source workbook changed; recheck the original question')
        q, mass, duration, dt, extent = E_CHARGE, MP, .024, .001, .0065
        electric = mass*G/q
        sources = (UniformElectricField((electric,0.,0.)),)
        note = ('IIT-JEE 2012 archive, workbook p15 Q2: proton initially at rest midway in a 1 cm vertical plate gap; 45-degree initial acceleration. Uniform-gap field only; preview ends before either plate. Rounded m_p=1.67e-27 kg, e=1.6e-19 C and g=10 m/s^2; duration illustrative.')
        anchor.update(kind='source-backed numerical realization',pdf_file=source.name,pdf_page=15,
                      printed_question='Archives Single Correct Q2 (IIT-JEE 2012)',
                      sha256=PDF_SHA,plate_gap_m=.01,voltage_v=electric*.01,
                      approximation='uniform imposed field; no plate induction, fringing, impact or radiation',
                      additional_choices='rounded constants, axis direction and finite preview duration')
    elif case == 'oblique-launch':
        q,mass,duration,dt,extent=1e-6,2e-6,.24,.01,.5
        sources=(UniformElectricField((6.,0.,0.)),)
        velocity=(.4,1.4,0.)
        note='Teacher hypothetical: oblique launch in independent horizontal uniform E and downward uniform gravity.'
    elif case == 'electric-support':
        q,mass,duration,dt,extent=1e-6,1e-6,.2,.01,.15
        sources=(UniformElectricField((0.,10.,0.)),)
        note='Teacher hypothetical: upward electric force exactly balances gravity; rest persists without a guide or hidden support.'
    elif case == 'gravity-only':
        q,mass,duration,dt,extent=1e-6,2e-6,.24,.01,.5
        sources=(UniformElectricField((0.,0.,0.)),)
        velocity=(1.,1.5,0.)
        note='Teacher hypothetical: electric field is zero. The same charged-particle model reduces to ordinary projectile motion under gravity.'
    elif case == 'point-plus-gravity':
        q,mass,duration,dt,extent=1e-9,1e-6,.2,.01,.4
        sources=(FixedPointField(2e-9,(-.25,0.,0.)),)
        labels=(('charge:held','Held point charge'),)
        origin,velocity=(0.,.05,0.),(.05,.03,0.)
        gravity=UniformGravity((0.,-.4,0.))
        note='Teacher hypothetical: a held point charge supplies a nonuniform electric force, composed with independently chosen uniform gravity.'
    elif case == 'ring-plus-gravity':
        q,mass,duration,dt,extent=1e-9,1e-6,.3,.01,.4
        sources=(UniformRingAxisField(2e-9,.1,axis=(0.,1.,0.)),)
        labels=(('ring:held','Held charged ring'),)
        origin,velocity=(0.,.2,0.),(0.,-.05,0.)
        gravity=UniformGravity((0.,-.1,0.))
        note='Teacher hypothetical: nonuniform ring-axis electric force plus collinear uniform gravity. No off-axis ring field or artificial guide is used.'
    else:
        raise ValueError('unknown M21e teaching arrangement')
    spec=TestChargeMotionSpec(PrescribedField(sources),PointCharge(q,origin),mass,
        velocity,duration,dt,plane_normal=(0.,0.,1.),length_scale_m=extent,
        source_guard_m=1e-6,gravity=gravity)
    scenario=PrescribedScenario('m21e-'+case,spec,labels,note)
    return scenario,anchor,extent


def make_bundle(case):
    scenario,anchor,extent=build_case(case)
    bundle=sample_scenario_viewer(scenario,visual_frames=13,resolution=17,
                                 profile_samples=101,extent_m=extent)
    for frame in bundle['frames']:
        frame['notes'].append(scenario.provenance)
    return scenario,bundle,anchor


def export_case(case, output):
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    if case=='mass-revision':
        base,anchor,extent=build_case('oblique-launch')
        revision=revise_scenario(base,scenario_id='m21e-double-mass',note='Teacher variation: double mass; electric field, charge and gravity unchanged.',motion_changes={'mass_kg':2*base.spec.mass_kg})
        kw=dict(visual_frames=13,resolution=17,profile_samples=101,extent_m=extent)
        left,right=(sample_scenario_viewer(s,**kw) for s in (base,revision))
        labels=dict(baseline_title='Original mass',variation_title='Doubled mass',
                    baseline_provenance=base.viewer_provenance,variation_provenance=revision.viewer_provenance)
        html=render_scenario_comparison(left,right,title='M21e · Electric and gravity mass dependence',**labels)
        audit=dict(baseline=base.as_dict(),variation=revision.as_dict(),
                   comparison=prepare_scenario_comparison(left,right,**labels),
                   baseline_bundle=left,variation_bundle=right)
    else:
        scenario,bundle,anchor=make_bundle(case)
        title=('M21e · IIT-JEE 2012 proton: electric force and gravity'
               if case=='jee2012-proton' else 'M21e · '+case.replace('-',' ').title())
        html=render_html(bundle,title=title)
        audit=dict(scenario=scenario.as_dict(),source_anchor=anchor,bundle=bundle)
    output.write_text(html,encoding='utf-8')
    output.with_suffix('.json').write_text(json.dumps(audit,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    return audit


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--case',choices=CASES,default=CASES[0])
    ap.add_argument('--output',type=Path,default=Path('output/m21e-proton.html'))
    args=ap.parse_args();audit=export_case(args.case,args.output)
    print(args.case, args.output)
    print('Same prescribed-field integrator and shared viewer; independent electric and gravitational force/energy terms.')
    if args.case=='jee2012-proton': print('Source-required gap voltage (V):',audit['source_anchor']['voltage_v'])

if __name__=='__main__':main()
