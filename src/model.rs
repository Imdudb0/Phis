use crate::net::Net;
use crate::percept::Percept;
use crate::rng::Rng;
use crate::search::{self, Budget, Outcome, Step};
use std::path::Path;
#[derive(Clone, Debug)]
pub struct Model {
    pub net: Net,
}
#[derive(Clone, Debug)]
pub struct Answer {
    pub percept: Percept,
    pub outcome: Outcome,
}
pub fn seed(eff: &[u8]) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for &b in eff {
        h = (h ^ u64::from(b)).wrapping_mul(0x0000_0100_0000_01b3);
    }
    h ^ eff.len() as u64
}
impl Model {
    pub fn new(net: Net) -> Model {
        Model { net }
    }
    pub fn to_bytes(&self) -> Vec<u8> {
        self.net.to_bytes()
    }
    pub fn from_bytes(b: &[u8]) -> Result<Model, String> {
        Net::from_bytes(b).map(Model::new)
    }
    pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<(), String> {
        let p: &Path = path.as_ref();
        self.net.save(p).map_err(|e| format!("{} : {e}", p.display()))
    }
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Model, String> {
        let p: &Path = path.as_ref();
        Net::load(p)
            .map(Model::new)
            .map_err(|e| format!("{} : {e}", p.display()))
    }
    pub fn ask(&self, eff: &[u8]) -> Answer {
        self.ask_with(eff, &mut |_: &Step| {})
    }
    pub fn ask_with(&self, eff: &[u8], on_step: &mut dyn FnMut(&Step)) -> Answer {
        let percept = Percept::new(eff);
        let mut rng = Rng::new(seed(eff));
        let budget = Budget::measured(&percept);
        let outcome = search::run(&self.net, &percept, &mut rng, budget, on_step);
        Answer { percept, outcome }
    }
    pub fn ask_file<P: AsRef<Path>>(
        &self,
        path: P,
        on_step: &mut dyn FnMut(&Step),
    ) -> Result<Answer, String> {
        let p: &Path = path.as_ref();
        let eff = std::fs::read(p).map_err(|e| format!("{} : {e}", p.display()))?;
        Ok(self.ask_with(&eff, on_step))
    }
}