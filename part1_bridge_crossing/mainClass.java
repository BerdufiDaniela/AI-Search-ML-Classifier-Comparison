import java.util.ArrayList;
import java.util.HashMap;
import java.util.Scanner;

public class mainClass {

    public static void main(String[] args){
        //N=ari8mos oikogeneias
        int N;
        //maxtime=megistos xronos gia na diasxixei h oikogeneia thn gefhra
        int maxtime;
        Scanner in = new Scanner(System.in);
        System.out.println("How many family members are? (Give a number)");
        N=in.nextInt();
        //pinakas t me antikeimena FamMember pou exei ws stoixeia to melos oikogeneias kai ton xrono pou kanei gia na diasxisei thn gefhra
        FamMember t[]= new FamMember[N];
        //time=xronos gia na diasxisei ka8e melos thn gefhra
        int time;
        //HashMap <String,Integer> listinfo = new HashMap <String,Integer>();
        //pinakas pou boi8aei na onomasoume ta meloi mias oikogeneias
        String[] m={"A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z"};
        for(int i=0;i<N;i++){
            //dhmiourgw antikeimeno FamMember
            FamMember fm= new FamMember();
            //dinw sto kaue melow onomasia
            fm.setMember(m[i]);
            System.out.printf("Give the time that takes the family member %s to cross the bridge : ",m[i]);
            time=in.nextInt();
            //listinfo.put(m[i],time);
            //dinw se ka8e melos xrono pou 8elei gia na perasei gefhra
            fm.setTime(time);
            //bazw to antikeimeno ston pinaka mou
            t[i]=fm; 
        }
        
        System.out.printf("Give the maximum time that should take the family to cross the bridge : ");
        //8etw ton megisto xrono gia na perasei oloi h oikogeneia thn gefhra
        maxtime=in.nextInt();
        //gia na metrame ton xrono toy susthmatos
        long startTime = System.currentTimeMillis();
        ////////////////////////////////////////////////////////////////////////
        //ta3inomw ton pinaka me tous xronous apo ka8e melos
        int i,key,j;
        FamMember f;
        for(i=1;i<N;i++){
            f=t[i];
            key=t[i].getTime();
            j=i-1;
            while(j>=0 && t[j].getTime()>key){
                t[j+1]=t[j];
                j=j-1;
            }
            t[j+1]=f;
        }
        ////////////////////////////////////////////////////////////////////////

        HashMap<String,Integer> listinfo=new HashMap <String,Integer>();
		for(int k =0;k<N;k++){
			listinfo.put(t[k].getMember(),t[k].getTime());
		}
        State st =new State();
        st.addelement("");
        State prev=new State ();
        prev.addelement("");
        State p= new State();
        int crosstime=0;
        int timestop=0;
        //st pou einai h katastash pou deixnei poioi emeinan apenanti
        while(st.isFinal(t,N)==false){
            //lista pou 8a apo8ikeusw ta paidia ths ka8e katastashs
            ArrayList<State> lis=new ArrayList<>();
            lis=st.getChildren(t,N);
            //minol metablhth gia na brw thn mikroterh euretikh
            int minol=1000000000;
            //sysxetish kostous euretikhs me katastash pou deixnei poioi exoun mhnei apenanti
            HashMap<Integer,State> statemininfo=new HashMap<Integer,State>();
            //sysxetish pou deixnei kostos euretikhs me to poioi akribws perasan kai gyrisan
            HashMap<Integer,State> statecrossinfo=new HashMap<Integer,State>();
            //lista gia na apo8hkeusw katastaseis me to poioi akribvs phgan kai poioi gurisan
            ArrayList<State> crosslist=new ArrayList<>();
            //gia ka8e paidi dhladh pi8ano zeugarh pou 8a perasei apenanti
            for (int k=0;k<lis.size();k++){
                //dhmiourgw kainourgia katastatsh gia na thn a3iologhsw
                State stat=new State();
                State stwho=new State();
                //pros8etw ola ta stoixeia pou exoun perasei hdh apenanti ektos apo to ""
                for(int h=0;h<st.getListmem().size();h++){
                    if(st.getListmem().get(h)!=""){
                        stat.addelement(st.getListmem().get(h));
                    }
                }
                //pros8etw kai to prothnomeno zeugarh dhladh to paidi pou prothne h sunarthsh
                stat.addelement(lis.get(k).getListmem().get(0));
                stat.addelement(lis.get(k).getListmem().get(1));
                //ftiaxnoume katastash me ta meloi pou 8a epistrepsoune kai 8a gurisoun
                stwho.addelement(lis.get(k).getListmem().get(0));
                stwho.addelement(lis.get(k).getListmem().get(1));
                //psaxnw to elaxisto xrono gia na gyrisei pisw me to fanari
                int min=listinfo.get(stat.getListmem().get(0));
                int indexmin=0;
                String minfam=stat.getListmem().get(0);
                for(int h=1;h<stat.getListmem().size();h++){
                    if(listinfo.get(stat.getListmem().get(h))<min){
                        min=listinfo.get(stat.getListmem().get(h));
                        indexmin=h;
                        minfam=stat.getListmem().get(h);
                    }
                }
                //psaxnw ton megalutero xrono
                int max=listinfo.get(stat.getListmem().get(0));
                for(int h=1;h<stat.getListmem().size();h++){
                    if(listinfo.get(stat.getListmem().get(h))>max){
                        max=listinfo.get(stat.getListmem().get(h));
                    }
                }
                //8etw to g
                int g=0;
                //an den einai telikh katastash to g einai o xronos pou 8a kanoun na pane mazi kai o mikroteros xronos pou 8a gurisei apo
                //autous pou exoun hdh perasei
                if(!(stat.isFinal(t,N))){
                    g=max+min;
                    //sthn katstash me autous pou 8a pane kai 8a gurisoun pros8etw to mikrotero pou 8a gurisei
                    //ama den einai telikh katastash
                    stwho.addelement(minfam);
                //alliws 8a ginei telikh katastash opte den gyrnaei kanenas
                }else{
                    g=max;
                }
                //pros8etw sthn lista tis katastaseis me to poioi akribvs 8a pane kai 8a gurisoun
                crosslist.add(stwho);
                //mono ama den einai telikh katastash na gurisei o mikroteros xronos
                if(stat.isFinal(t,N)==false){
                    stat.removel(indexmin);
                }
                //upologizw euretikh
                stat.evaluate(t,N,g);
                //bazw sto hashmap xrono euretikhs kai katastash pou antistoixei
                statemininfo.put(stat.getF(),stat);
                //bazw sto HashMap xrono euretikhs kai katastash me to poioi akribvs phgan kai poioi gyrisan
                statecrossinfo.put(stat.getF(),stwho);
                //briskw to mikrotero xrono euretikhs
                if(stat.getF()<minol){
                    minol=stat.getF();
                }
            }
            ///////////////////////////////////////////////////////////
            //adeiazw ka8e fora to st kai pros8etw thn katastash me to posous exoun perasei kai exoun meinei apo to stat
            //giati alliws afairv mikrotero xrono apo stat alla oxi apo st kai mono prosti8ontai katastaseis
            st.getListmem().clear();
            //pros8etw sthn katastash st ta stoixeia ths katastashs stat pou einai ta meloi pou emeinan epenanti ka8e fora
            
            for(int h=0;h<statemininfo.get(minol).getListmem().size();h++){
                st.addelement(statemininfo.get(minol).getListmem().get(h));
            }
            
            //ana8esh pateradwn//////////////////////////////////////////
            for(int w=0;w<crosslist.size();w++){
                crosslist.get(w).setFather(prev);
            }
            //twra 8a ginei pateras autos me to mikrotero kostos euretikhs
            p=statecrossinfo.get(minol);
            prev=p;
            //upologizoume ton xrono pou kanoun ta meloi gia na pane kai na gurisoune ama den einai telikh katastash
            if(statecrossinfo.get(minol).getListmem().size()==3){
                timestop=timestop+listinfo.get(statecrossinfo.get(minol).getListmem().get(1))+listinfo.get(statecrossinfo.get(minol).getListmem().get(2));
            }
            //upologizoume ton xrono pou kanoun ta meloi gia na perasoun ama eimaste sthn telikh katastash
            if(statecrossinfo.get(minol).getListmem().size()==2){
                timestop=timestop+listinfo.get(statecrossinfo.get(minol).getListmem().get(1));
            }
            //ama o xronos autvn pou exoun perasei einai megaluteros apo ton megisto xrono pou exei dwsei o xrhsths
            //alla den exoume ftasei akomh se telikh katastash tote stamatame
            if(timestop>maxtime){
                break;
            }

        }//end while
        State last=new State();
        last=p;
        //apo8hkeuw thn diadromh se lista gia na mporesv na thn parv me thn svsth seira kai oxi anapoda
        ArrayList <State> path=new ArrayList<>();
        path.add(last);
        while(!(p.getListmem().contains(""))) // if father is null, then we are at the root.
        {
            path.add(p.getFather());
            p=p.getFather();
            
        }
        long endTime = System.currentTimeMillis();
        long executionTime = endTime - startTime;
        //emfanizw diadromh
        if(timestop<=maxtime){
            System.out.println("");
            System.out.println("******************** THE PATH IS ***************************");
            for(int u=path.size()-1;u>=0;u--){
                path.get(u).print(path.get(u));
            }
            System.out.printf("The smallest time of following this path is : %s%n",timestop);
            System.out.println("************************************************************");
        }else{
            System.out.println("");
            System.out.println("**************** SORRY THERE IS NO PATH *******************");
            System.out.printf("The family cannot cross the bridge in less time than: %d%n",maxtime);
            System.out.println("************************************************************");
        }
        System.out.println("Execution time: " + executionTime + " milliseconds");
       
    }
    
}
