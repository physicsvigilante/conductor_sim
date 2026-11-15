"""M23a: one dipole observer, multiple independently authored source systems."""
import argparse
from math import pi
from pathlib import Path

from conductor_sim.dipoles import (DipoleModel, RigidDipole, revise_dipole_model,
    dipole_work, observe_dipole, orient_dipole)
from conductor_sim.dipole_viewer import write_dipole_explorer
from conductor_sim.prescribed_fields import (FixedPointField, UniformElectricField,
    UniformRodField, UniformSphereField)
from conductor_sim.structural_scenarios import NamedSource

CASES = ('workbook-q47', 'point-field', 'mixed-sources', 'revised-field')


def make_model(case):
    dipole = RigidDipole(1e-9, .04)
    if case in ('workbook-q47', 'revised-field'):
        sources = (NamedSource('field:uniform', 'Uniform applied E', UniformElectricField((120.,0.,0.))),)
        provenance = ('Workbook physical PDF p5, Q47: initially parallel dipole; final acute, '
            'perpendicular, obtuse, antiparallel directions. Source gives NO numerical angles or '
            'q/length/E. Here 40°, 90°, 115°, 180°, q=1 nC, separation=.04 m and E=120 V/m are '
            'illustrative. Increasing change in energy: i < ii < iii < iv; all changes are positive.')
    elif case == 'point-field':
        sources = (NamedSource('point:held', 'Held positive charge', FixedPointField(2e-9,(.18,.10,0.))),)
        dipole = RigidDipole(1e-9,.04,center_m=(.01,0.,0.),pivot_m=(-.02,-.01,0.))
        provenance = ('Teacher hypothetical: finite dipole in a nonuniform held-point field; '
            'explicit off-centre pivot. Every numerical value is authored. Endpoint forces and '
            'centre displacement are real; no local p×E approximation or time evolution.')
    elif case == 'mixed-sources':
        sources = (
            NamedSource('rod:held','Held uniform rod',UniformRodField(-.8e-9,(-.15,.20,0.),(.10,.24,0.))),
            NamedSource('sphere:volume','Fixed volume sphere',UniformSphereField(1e-9,.06,(-.23,.05,0.))),
            NamedSource('field:background','Applied uniform E',UniformElectricField((20.,40.,0.))))
        provenance = ('Preselected teacher-hypothetical transfer: finite rod + fixed charged volume '
            'sphere + uniform E. One unchanged finite-dipole observer; no case-specific solver.')
    else:
        raise ValueError('unknown example')
    model = DipoleModel(dipole, sources, 'm23a-'+case, provenance)
    if case == 'revised-field':
        base, _ = make_model('workbook-q47')
        model, audit = revise_dipole_model(base, model_id='m23a-revised-field',
            note='Double the external electric field; retain q, separation and source identity.',
            source_changes={'field:uniform': {'field_v_m': (240.,0.,0.)}})
        return model, audit
    return model, None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--case', choices=CASES, default=CASES[0])
    ap.add_argument('--output', type=Path, default=Path('output/m23a-dipole.html'))
    a = ap.parse_args()
    model, revision = make_model(a.case)
    angles = tuple(i*pi/36 for i in range(73))
    path = write_dipole_explorer(model, angles, a.output, title='M23a · '+a.case)
    if revision:
        import json
        path.with_suffix('.revision.json').write_text(json.dumps(revision,indent=2)+'\n',encoding='utf-8')
    print(path)
    if a.case == 'workbook-q47':
        for angle in (40,90,115,180):
            print(angle, 'illustrative degrees', dipole_work(model,0.,angle*pi/180))


if __name__ == '__main__':
    main()
