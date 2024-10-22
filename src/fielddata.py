class Field_Data:
    #class to store information for experimental phase fields
    #TODO streamline grabbing of data from mpds
    def __init__(self,phase_field):
        if phase_field==['Li','Al','B','O']:
            self.pure_phases={
                "Li 2 Al B 5 O 10":[2,1,5,10],
                "Li B 3 O 5":[1,0,3,5],
                "Li 2 B 4 O 7":[2,0,4,7],
                "Li B O 2":[1,0,1,2],
                "Li 6 B 4 O 9":[6,0,4,9],
                "Li 4 B 2 O 5":[4,0,2,5],
                "Li 3 B O 3":[3,0,1,3],
                "B 2 O 3":[0,0,2,3],
                "Li 2 O":[2,0,0,1],
                "Li Al B 2 O 5":[1,1,2,5],
                "Li 3 Al B 2 O 6":[3,1,2,6],
                "Li 2 Al B O 4":[2,1,1,4],
                "Li Al 7 B 4 O 17":[1,7,4,17],
                "Li 2.46 Al 0.18 B O 3":[2.46,0.18,1,3],
                "Al 4.91 B 1.09 O 9":[0,4.91,1.09,9],
                "Al 2 O 3":[0,2,0,3],
                "Li Al O 2":[1,1,0,2]}
            self.triangles=[
                ["B 2 O 3","Li B 3 O 5","Li Al 7 B 4 O 17"],
                ["B 2 O 3","Li Al 7 B 4 O 17","Al 4.91 B 1.09 O 9"],
                ["Li Al 7 B 4 O 17","Al 4.91 B 1.09 O 9","Al 2 O 3"],
                ["Al 2 O 3","Li Al 7 B 4 O 17","Li Al B 2 O 5"],
                ["Al 2 O 3","Li Al B 2 O 5","Li 3 Al B 2 O 6"],
                ["Al 2 O 3","Li 3 Al B 2 O 6","Li 2 Al B O 4"],
                ["Al 2 O 3","Li 2 Al B O 4","Li Al O 2"],
                ["Li Al O 2","Li 2 O","Li 3 B O 3"],
                ["Li Al O 2","Li 2.46 Al 0.18 B O 3","Li 2 Al B O 4"],
                ["Li Al O 2","Li 2.46 Al 0.18 B O 3","Li 3 B O 3"],
                ["Li 2.46 Al 0.18 B O 3","Li 2 Al B O 4","Li 6 B 4 O 9"],
                ["Li 2.46 Al 0.18 B O 3","Li 6 B 4 O 9","Li 4 B 2 O 5"],
                ["Li 2.46 Al 0.18 B O 3","Li 4 B 2 O 5","Li 3 B O 3"],
                ["Li 2 Al B O 4", "Li 3 Al B 2 O 6","Li 6 B 4 O 9"],
                ["Li 3 Al B 2 O 6","Li 6 B 4 O 9","Li B O 2"],
                ["Li 3 Al B 2 O 6","Li B O 2","Li Al B 2 O 5"],
                ["Li Al B 2 O 5","Li B O 2","Li 2 B 4 O 7"],
                ["Li Al B 2 O 5","Li 2 B 4 O 7","Li 2 Al B 5 O 10"],
                ["Li Al B 2 O 5","Li 2 Al B 5 O 10","Li Al 7 B 4 O 17"],
                ["Li 2 Al B 5 O 10","Li 2 B 4 O 7","Li B 3 O 5"],
                ["Li 2 Al B 5 O 10","Li B 3 O 5","Li Al 7 B 4 O 17"]]
        else:
            raise Exception(
                "Data is not available for the phase field" + str(phase_field))

