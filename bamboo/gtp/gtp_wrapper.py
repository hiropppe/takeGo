import sys
import multiprocessing

from bamboo.gtp import gtp

from bamboo.gtp.gtp_connector import GTPGameConnector

AXIS = 'abcdefghjklmnopqrst'
RESULT = 'DBW'


def run_gnugo(sgf_file_name, command):
    from distutils import spawn
    if spawn.find_executable('gnugo'):
        from subprocess import Popen, PIPE
        p = Popen(['gnugo', '--chinese-rules', '--mode', 'gtp', '-l', sgf_file_name],
                  stdout=PIPE, stdin=PIPE, stderr=PIPE)
        out_bytes = p.communicate(input=command)[0]
        return out_bytes.decode('utf-8')[2:]
    else:
        return ''


class ExtendedGtpEngine(gtp.Engine):

    recommended_handicaps = {
        2: "D4 Q16",
        3: "D4 Q16 D16",
        4: "D4 Q16 D16 Q4",
        5: "D4 Q16 D16 Q4 K10",
        6: "D4 Q16 D16 Q4 D10 Q10",
        7: "D4 Q16 D16 Q4 D10 Q10 K10",
        8: "D4 Q16 D16 Q4 D10 Q10 K4 K16",
        9: "D4 Q16 D16 Q4 D10 Q10 K4 K16 K10"
    }

    def call_gnugo(self, sgf_file_name, command):
        try:
            pool = multiprocessing.Pool(processes=1)
            result = pool.apply_async(run_gnugo, (sgf_file_name, command))
            output = result.get(timeout=10)
            pool.close()
            return output
        except multiprocessing.TimeoutError:
            pool.terminate()
            # if can't get answer from GnuGo, return no result
            return ''

    def cmd_time_settings(self, arguments):
        try:
            main_time, byo_yomi_time, byo_yomi_stone = [int(t) for t in arguments.split()]
        except ValueError:
            raise ValueError('Invalid time_settings input: {}'.format(arguments))
        self._game.set_time(main_time, byo_yomi_time, byo_yomi_stone)

    def cmd_time_left(self, arguments):
        try:
            color, time, stone = arguments.split()
            self._game.set_time_left(color, (int)(time), (int)(stone))
        except ValueError:
            raise ValueError('Invalid time_left input: {}'.format(arguments))

    def cmd_place_free_handicap(self, arguments):
        try:
            number_of_stones = int(arguments)
        except Exception:
            raise ValueError('Number of handicaps could not be parsed: {}'.format(arguments))
        if number_of_stones < 2 or number_of_stones > 9:
            raise ValueError('Invalid number of handicap stones: {}'.format(number_of_stones))
        vertex_string = ExtendedGtpEngine.recommended_handicaps[number_of_stones]
        self.cmd_set_free_handicap(vertex_string)
        return vertex_string

    def cmd_set_free_handicap(self, arguments):
        vertices = arguments.strip().split()
        moves = [gtp.parse_vertex(vertex) for vertex in vertices]
        self._game.place_handicaps(moves)

    def cmd_final_score(self, arguments):
        sgf_file_name = self._game.get_current_state_as_sgf()
        return self.call_gnugo(sgf_file_name, 'final_score\n')

    def cmd_final_status_list(self, arguments):
        sgf_file_name = self._game.get_current_state_as_sgf()
        return self.call_gnugo(sgf_file_name, 'final_status_list {}\n'.format(arguments))

    def cmd_load_sgf(self, arguments):
        pass

    def cmd_save_sgf(self, arguments):
        pass

    # def cmd_kgs_genmove_cleanup(self, arguments):
    #     return self.cmd_genmove(arguments)


def run_gtp(gtp_game, inpt_fn=None, name="Gtp Player", version="0.0"):
    gtp_engine = ExtendedGtpEngine(gtp_game, name, version)
    if inpt_fn is None:
        inpt_fn = raw_input

    sys.stderr.write("GTP engine ready\n")
    sys.stderr.flush()
    while not gtp_engine.disconnect:
        inpt = inpt_fn()
        # handle either single lines at a time
        # or multiple commands separated by '\n'
        try:
            cmd_list = inpt.split("\n")
        except:
            cmd_list = [inpt]
        for cmd in cmd_list:
            engine_reply = gtp_engine.send(cmd)
            sys.stdout.write(engine_reply)
            sys.stdout.flush()
